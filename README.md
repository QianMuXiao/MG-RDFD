# MG-RDFD

This repository contains the research implementation of **Memory-Guided Random-Direction Feature Disentanglement (MG-RDFD)** for multi-phase contrast-enhanced MRI translation.

The code includes the main encoder-decoder model, random-direction content/style sampling modules, segmentation-guided memory bank, paired MRI dataset loader, and training pipeline used in our study. Due to dataset access requirements, environment differences, and hardware-dependent training settings, this repository is intended as a reference implementation rather than a one-command reproduction package.

## Code Structure

```text
.
├── Models_v2.py              # Autoencoder-based translation backbone
├── Sample.py                 # Random-direction content/style sampling modules
├── Memory_pair_v7.py          # Segmentation-guided memory bank
├── mri_dataset_v4.py          # Paired multi-phase MRI slice dataset
├── loss_fn.py                 # Training losses
├── train_v5.py                # Distributed training script
├── training_utils.py          # Gradient synchronization and validation utilities
├── tests/                     # CPU regression tests
└── lesion_patient_list.txt    # Patient-level lesion category list used for splitting
```

## Environment

The code was developed with PyTorch and common medical image processing libraries. A typical environment should include:

```text
python
torch
torchvision
monai-generative
SimpleITK
scipy
numpy
tqdm
tensorboardX
pytorch-msssim
lpips
xformers
```

The attention blocks can use xFormers/FlashAttention when available. If xFormers is not compatible with the local GPU, CUDA, or PyTorch version, disable the flash-attention path in the model configuration before training or inference.

## Dataset

This work uses the LLD-MMRI dataset for multi-phase liver lesion MRI analysis. The dataset is not redistributed in this repository. Please refer to the official dataset repository for access and usage terms:

- LLD-MMRI dataset: https://github.com/LMMMEng/LLD-MMRI-Dataset

After preprocessing the original 3D volumes into paired 2D slices, the dataset loader expects the following structure:

```text
LLD-MMRI/
├── lesion_patient_list.txt
│
└── 2d_mri_body_dataset_mutil_phase_corp_body_v1/
    ├── pre/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    ├── c_a/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    ├── c_v/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    ├── delay/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    ├── totalseg/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    ├── tumor/
    │   ├── MR-391135_10.nii.gz
    │   ├── MR-391135_11.nii.gz
    │   └── ...
    │
    └── body/
        ├── MR-391135_10.nii.gz
        ├── MR-391135_11.nii.gz
        └── ...
```

Each slice is stored as a 2D `.nii.gz` file named as `{patient_id}_{slice_index}.nii.gz`. The same filename should exist in all required phase and mask folders so that the loader can pair slices by `(patient_id, slice_index)`.

The `data_path` argument should point to:

```text
LLD-MMRI/2d_mri_body_dataset_mutil_phase_corp_body_v1
```

The `lesion_patient_file` argument should point to:

```text
LLD-MMRI/lesion_patient_list.txt
```

The dataset class performs lesion-stratified patient-level train/validation splitting internally. Separate `train/` and `val/` folders are therefore not required.

## Implementation corrections

The September 2026 maintenance update fixes the following issues in the released
implementation while preserving the model architecture, loss weights and default
training hyperparameters:

- Patient filename matching includes the underscore delimiter, so an ID cannot
  accidentally select a different patient whose ID shares its prefix. Paired
  sample keys are sorted before the existing seeded shuffle to keep dataset
  indices consistent across distributed processes.
- The generator is accessed through explicit encode/decode methods in the
  training loop. These calls bypass DDP's forward path, so its gradients are now
  explicitly averaged across ranks before each optimizer step. The content and
  style samplers retain their existing DDP synchronization.
- Validation assigns each sample to exactly one rank, without padding the last
  shard with duplicate samples. Validation-only sampler calls use the underlying
  modules, allowing different numbers of validation batches on different ranks.
- PSNR is calculated per image and then averaged over samples. An identical
  prediction returns a tensor containing positive infinity. This avoids both
  batch-grouping dependence and the former Python-float `.item()` error.
- Resumed training uses the saved next epoch and best validation PSNR, instead
  of restarting epoch numbering and warm-up from zero.
- The default memory constructor uses the defined V7 label list. The public
  autoencoder `forward` method and its example use the existing A/B raw decoders.

These corrections do not retroactively change existing checkpoints or reported
results. Keep the commit ID and training configuration with each new experiment.
The original published source remains available in the Git history; the last
commit before these fixes is `2eee0ed11c8116bc1db3341be1776f7252783c65`.

### Regression tests

With the dataset and model dependencies installed, run the CPU regression suite:

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic data and small models, including multiple CPU processes
with the Gloo backend. They do not download datasets or pretrained weights.
They cover the corrected interfaces and distributed behavior; they are not a
replacement for a complete training run or a reproduction of the paper's scores.

## Training Notes

Before launching training, set the task-specific paths and phases in `train_v5.py`, including:

```python
phase_1 = "pre"
phase_2 = "c_a"  # or "c_v"
data_path = "/path/to/LLD-MMRI/2d_mri_body_dataset_mutil_phase_corp_body_v1"
lesion_patient_file = "/path/to/LLD-MMRI/lesion_patient_list.txt"
```

The script uses PyTorch distributed training utilities. For a single GPU run, a typical launch pattern is:

```bash
torchrun --nproc_per_node=1 train_v5.py
```

Please adjust batch size, learning rate, number of workers, save paths, and attention settings according to your hardware and software environment.

## Citation

If you use this code, please cite the corresponding MG-RDFD paper. Citation information will be updated after the proceedings metadata is available.
