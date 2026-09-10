import ast
import json
import math
import os
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
import uuid

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from training_utils import average_module_gradients, DistributedEvalSampler, mean_sample_psnr


class _ToyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.01, 0.37, 37, dtype=torch.float64))
        self.optional = nn.Parameter(torch.tensor([0.03], dtype=torch.float64))
        self.unused = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.weight32 = nn.Parameter(torch.tensor([0.02], dtype=torch.float32))

    def encode(self, x, include_optional):
        value = x @ self.weight + x[:, 0] * self.weight32.double()
        if include_optional:
            value = value + self.optional * x[:, 0]
        return value.unsqueeze(1)

    def forward(self, x, include_optional):
        return self.encode(x, include_optional)


def _distributed_worker(rank, port, destination):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if os.name == 'nt':
        os.environ.setdefault('GLOO_DEVICE_TRANSPORT', 'UV')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2,
                            init_method=f'tcp://127.0.0.1:{port}?use_libuv=0')
    try:
        generator = DDP(_ToyGenerator(), find_unused_parameters=True)
        sampler_module = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        nn.init.constant_(sampler_module.weight, 0.1)
        sampler = DDP(sampler_module)
        ref_generator = _ToyGenerator()
        ref_sampler = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        nn.init.constant_(ref_sampler.weight, 0.1)
        opt = torch.optim.SGD(list(generator.parameters()) + list(sampler.parameters()), lr=0.01, momentum=0.9)
        ref_opt = torch.optim.SGD(list(ref_generator.parameters()) + list(ref_sampler.parameters()), lr=0.01, momentum=0.9)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            ref_opt.zero_grad(set_to_none=True)
            x = torch.full((1, 37), rank + 1.0, dtype=torch.float64)
            sampler(generator.module.encode(x, rank == 0)).square().mean().backward()
            # Small bucket forces a single large parameter across several collectives.
            average_module_gradients(generator.module, bucket_cap_mb=32 / (1024 * 1024))
            ref_loss = sum(ref_sampler(ref_generator.encode(
                torch.full((1, 37), r + 1.0, dtype=torch.float64), r == 0)).square().mean()
                           for r in range(2)) / 2
            ref_loss.backward()
            assert generator.module.unused.grad is None
            for actual, expected in zip(generator.module.parameters(), ref_generator.parameters()):
                if expected.grad is None:
                    assert actual.grad is None
                else:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-10)
            torch.testing.assert_close(sampler.module.weight.grad, ref_sampler.weight.grad, rtol=1e-10, atol=1e-10)
            opt.step()
            ref_opt.step()
            for actual, expected in zip(generator.module.parameters(), ref_generator.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-10)
            torch.testing.assert_close(sampler.module.weight, ref_sampler.weight, rtol=1e-10, atol=1e-10)

        # Validation can have unequal batch counts, including an empty rank.
        validation_results = []
        for count in (5, 1):
            dataset = list(range(count))
            shard = DistributedEvalSampler(dataset, shuffle=False)
            local = []
            with torch.no_grad():
                for batch in DataLoader(dataset, batch_size=2, sampler=shard):
                    values = batch.to(torch.float64).reshape(-1, 1)
                    sampler.module(values)  # Validation intentionally bypasses DDP.
                    local.extend(batch.tolist())
            all_rows = [None, None]
            dist.all_gather_object(all_rows, local)
            assert sorted(sum(all_rows, [])) == dataset
            validation_results.append(all_rows)
        if rank == 0:
            Path(destination).write_text(json.dumps({'passed': True, 'torch_version': torch.__version__,
                'backend': 'gloo', 'device': 'cpu', 'world_size': 2, 'steps': 2,
                'cases': ['mixed dtypes', 'bounded buckets', 'one-rank missing gradient',
                          'globally missing gradient', 'global-batch reference', 'DDP sampler not averaged twice',
                          'unequal validation batches', 'empty validation rank'],
                'validation_shards': validation_results}), encoding='utf-8')
    finally:
        dist.destroy_process_group()


class TrainingUtilitiesTest(unittest.TestCase):
    def test_psnr_is_batch_partition_invariant_and_keeps_infinity(self):
        target = torch.zeros(3, 1, 2, 2)
        prediction = torch.tensor([0.1, 0.01, 0.001]).reshape(3, 1, 1, 1).expand_as(target)
        whole = mean_sample_psnr(prediction, target)
        weighted = (mean_sample_psnr(prediction[:2], target[:2]) * 2 +
                    mean_sample_psnr(prediction[2:], target[2:])) / 3
        torch.testing.assert_close(whole, weighted)
        self.assertAlmostEqual(whole.item(), 40.0, places=5)
        self.assertTrue(math.isinf(mean_sample_psnr(target, target).item()))

    def test_eval_sampler_has_no_duplicates_or_missing_indices(self):
        for size in (0, 1, 5, 11):
            for world_size in (1, 2, 4):
                for shuffle in (False, True):
                    samplers = [DistributedEvalSampler(range(size), world_size, rank, shuffle, 42)
                                for rank in range(world_size)]
                    for epoch in (0, 3):
                        for sampler in samplers:
                            sampler.set_epoch(epoch)
                        shards = [list(sampler) for sampler in samplers]
                        self.assertEqual(sorted(sum(shards, [])), list(range(size)))
                        self.assertEqual([len(s) for s in samplers], [len(s) for s in shards])

    def test_resume_starts_at_saved_epoch_and_preserves_best_metric(self):
        # Execute the real function body without importing unrelated optional
        # training dependencies. Stop before model forward or optimizer work.
        source = (Path(__file__).resolve().parents[1] / 'train_v5.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_FDA_V2')
        namespace = {'torch': torch, 'is_master': lambda: False}
        exec(compile(ast.Module(body=[train], type_ignores=[]), 'train_v5.py', 'exec'), namespace)
        train_fn = namespace['train_FDA_V2']

        class StopBeforeFirstBatch(Exception):
            pass

        class RecordingSampler:
            def __init__(self):
                self.epochs = []
            def set_epoch(self, epoch):
                self.epochs.append(epoch)

        class Loader:
            def __iter__(self):
                raise StopBeforeFirstBatch()

        sampler = RecordingSampler()
        try:
            train_fn(train_loader=Loader(), train_sampler=sampler, cont_sampler=None, style_sampler=None,
                     val_loader=None, val_sampler=None, generator=nn.Identity(), disc_A=nn.Identity(),
                     disc_B=nn.Identity(), cont_disc=None, adv_loss=None, memory_module=nn.Identity(),
                     memory_list=[], opt_gen=None, opt_dis_A=None, opt_dis_B=None,
                     opt_cont_sampler=None, opt_style_sampler=None, Max_Epoch=10,
                     start_epoch=7, init_best_a2b_psnr=27.5)
        except StopBeforeFirstBatch as error:
            trace = error.__traceback__
            while trace.tb_frame.f_code is not train_fn.__code__:
                trace = trace.tb_next
            self.assertEqual(trace.tb_frame.f_locals['epoch_nums'], 7)
            self.assertEqual(trace.tb_frame.f_locals['best_a2b_psnr'], 27.5)
            self.assertEqual(sampler.epochs, [7])
        else:
            self.fail('Training did not reach the first resumed batch')

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU Gloo is unavailable')
    def test_two_rank_gradients_match_global_batch_and_validation_finishes(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        # Ordinary mkdir inherits Windows ACLs; TemporaryDirectory(mode=0700)
        # can prevent spawned sandbox workers from writing their result file.
        base = Path(tempfile.gettempdir()).resolve()
        tmp = base / ('mg_rdfd_cpu_' + uuid.uuid4().hex)
        tmp.mkdir()
        try:
            destination = str(Path(tmp) / 'result.json')
            try:
                mp.spawn(_distributed_worker, args=(port, destination), nprocs=2, join=True)
            except mp.ProcessRaisedException as error:
                # Only skip the identified Windows backend initialization failure;
                # gradient, coverage, or runtime assertion failures remain errors.
                if (os.name == 'nt' and 'init_process_group' in str(error)
                        and 'makeDeviceForHostname(): unsupported gloo device' in str(error)):
                    self.skipTest('This Windows PyTorch build cannot initialize a CPU Gloo device; run this test on Linux/WSL')
                raise
            result = json.loads(Path(destination).read_text(encoding='utf-8'))
            self.assertTrue(result['passed'])
            print('CPU_DISTRIBUTED_EVIDENCE=' + json.dumps(result))
        finally:
            if tmp.resolve().parent != base:
                raise RuntimeError('Unexpected test directory')
            shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
