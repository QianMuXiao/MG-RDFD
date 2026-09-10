"""Small training utilities; no changes to model architecture or loss weights."""
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def mean_sample_psnr(prediction, target):
    """Mean image PSNR for a [N, ...] batch in [0, 1]; exact matches stay inf."""
    mse = (prediction - target).square().flatten(1).mean(1)
    return (-10.0 * torch.log10(mse)).mean()


class DistributedEvalSampler(Sampler):
    """Shard indices without padding; validation forwards must not communicate."""
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        self.dataset = dataset
        self.num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        self.rank = dist.get_rank() if rank is None else rank
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError('Invalid evaluation rank or number of replicas')
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = range(len(self.dataset))
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self):
        return max(0, (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


@torch.no_grad()
def average_module_gradients(module, bucket_cap_mb=16):
    """Average dense grads after bypassing DDP.forward via custom module methods.

    Call once after backward and before step on every rank. Do not also call this
    for modules whose DDP forward already synchronizes their gradients. A missing
    local gradient contributes zero; globally unused parameters retain grad=None.
    Reduction buffers are bounded by bucket_cap_mb (a noncontiguous gradient may
    additionally need its own contiguous copy).
    """
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    if bucket_cap_mb <= 0:
        raise ValueError('bucket_cap_mb must be positive')
    groups = {}
    for parameter in module.parameters():
        if parameter.requires_grad:
            groups.setdefault((parameter.device, parameter.dtype), []).append(parameter)
    for (device, dtype), parameters in groups.items():
        # One collective per group also makes unequal None patterns safe.
        flags = torch.tensor(
            [[int(p.grad is not None), int(p.grad is not None and p.grad.is_sparse)]
             for p in parameters], device=device, dtype=torch.int32)
        dist.all_reduce(flags, op=dist.ReduceOp.SUM)
        states = flags.tolist()
        if any(sparse for _, sparse in states):
            raise ValueError('average_module_gradients supports dense gradients only')
        active = [p for p, (present, _) in zip(parameters, states) if present]
        if not active:
            continue
        element_size = active[0].element_size()
        capacity = max(1, int(bucket_cap_mb * 1024 * 1024) // element_size)
        capacity = min(capacity, sum(p.numel() for p in active))
        buffer = torch.empty(capacity, device=device, dtype=dtype)
        pending = []
        noncontiguous = []
        used = 0

        def flush():
            nonlocal used
            if used:
                reduced = buffer[:used]
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                reduced.div_(dist.get_world_size())
                for flat, start, length, offset in pending:
                    flat[start:start + length].copy_(reduced[offset:offset + length])
                pending.clear()
                used = 0

        for parameter in active:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter, memory_format=torch.contiguous_format)
            flat = parameter.grad.reshape(-1)
            if not parameter.grad.is_contiguous():
                noncontiguous.append((parameter, flat))
            start = 0
            while start < flat.numel():
                length = min(capacity - used, flat.numel() - start)
                buffer[used:used + length].copy_(flat[start:start + length])
                pending.append((flat, start, length, used))
                used += length
                start += length
                if used == capacity:
                    flush()
        flush()
        for parameter, flat in noncontiguous:
            parameter.grad.copy_(flat.view_as(parameter))
