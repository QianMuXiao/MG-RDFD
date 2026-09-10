"""CPU regressions for exact patient matching and deterministic pair order.

Run: python -m unittest discover -s tests -p test_mri_dataset.py -v
Uses the real dataset module and synthetic files only; no models or GPU execution.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import SimpleITK as sitk
import torch

from mri_dataset_v4 import PairMRIDataset


CHANNELS = ('pre', 'c_a', 'totalseg', 'tumor', 'body')
PATIENT_CLASSES = {
    '0': ('MR13762', 'MR137627'),
    '1': ('OTHER01', 'OTHER02', 'OTHER03', 'OTHER04'),
}
SLICE_IDS = ('11', '2', '7', '0')


def file_key(path):
    name = Path(path).name
    return tuple(name[:-7].split('_'))


def build_dataset(root, **kwargs):
    options = dict(data_path=str(root / 'images'),
                   lesion_patient_file=str(root / 'lesions.txt'),
                   split='train', val_ratio=0.2, image_size=(8, 8),
                   random_seed=42, phase_1='pre', phase_2='c_a')
    options.update(kwargs)
    return PairMRIDataset(**options)


def dump_order(root):
    return {split: [[Path(p).name for p in pair] for pair in
                    build_dataset(root, split=split).image_pairs]
            for split in ('train', 'val')}


class PairMRIDatasetTests(unittest.TestCase):
    def setUp(self):
        # Use ordinary mkdir so test fixtures inherit permissions on Windows.
        self.temp_parent = Path(tempfile.gettempdir()).resolve()
        self.root = self.temp_parent / ('mg_rdfd_dataset_test_' + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.remove_fixture)
        (self.root / 'lesions.txt').write_text(
            '\n'.join(line for category, patients in PATIENT_CLASSES.items()
                      for line in [f'病灶分类 {category}:',
                                   *(f'病人 ID: {patient}' for patient in patients)]) + '\n',
            encoding='utf-8')
        for channel in CHANNELS:
            folder = self.root / 'images' / channel
            folder.mkdir(parents=True)
            for patients in PATIENT_CLASSES.values():
                for patient in patients:
                    for slice_id in SLICE_IDS:
                        (folder / f'{patient}_{slice_id}.nii.gz').touch()

    def remove_fixture(self):
        resolved = self.root.resolve()
        if (resolved.parent != self.temp_parent or
                not resolved.name.startswith('mg_rdfd_dataset_test_')):
            raise RuntimeError(f'Refusing to remove unexpected fixture path: {resolved}')
        if resolved.exists():
            shutil.rmtree(resolved)

    def test_prefix_patients_do_not_leak_between_splits(self):
        train = build_dataset(self.root, split='train')
        val = build_dataset(self.root, split='val')
        train_keys = {file_key(pair[0]) for pair in train.image_pairs}
        val_keys = {file_key(pair[0]) for pair in val.image_pairs}
        self.assertIn('MR13762', train.patient_ids)
        self.assertIn('MR137627', val.patient_ids)
        self.assertFalse(set(train.patient_ids) & set(val.patient_ids))
        self.assertFalse(train_keys & val_keys)
        self.assertFalse({p for p, _ in train_keys} & {p for p, _ in val_keys})
        self.assertEqual({s for p, s in train_keys if p == 'MR13762'}, set(SLICE_IDS))
        self.assertEqual({s for p, s in val_keys if p == 'MR137627'}, set(SLICE_IDS))
        self.assertFalse(any(p == 'MR137627' for p, _ in train_keys))
        expected = {(p, s) for patients in PATIENT_CLASSES.values()
                    for p in patients for s in SLICE_IDS}
        self.assertEqual(train_keys | val_keys, expected)

    def test_all_five_channels_match_the_same_patient_and_slice(self):
        for split in ('train', 'val'):
            with self.subTest(split=split):
                ds = build_dataset(self.root, split=split)
                expected = {(p, s) for p in ds.patient_ids for s in SLICE_IDS}
                self.assertEqual(len(ds), len(expected))
                for channel_index, channel in enumerate(CHANNELS):
                    self.assertEqual({file_key(pair[channel_index]) for pair in ds.image_pairs},
                                     expected, channel)
                for pair in ds.image_pairs:
                    self.assertEqual(len(pair), 5)
                    self.assertEqual(len({file_key(path) for path in pair}), 1)
                    self.assertEqual(tuple(Path(path).parent.name for path in pair), CHANNELS)

    def test_sample_order_is_identical_across_python_hash_seeds(self):
        outputs = []
        for hash_seed in ('0', '12345'):
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = hash_seed
            process = subprocess.run(
                [sys.executable, '-B', str(Path(__file__).resolve()),
                 '--dump-order', str(self.root)],
                cwd=REPO_ROOT, env=env, text=True, encoding='utf-8',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=45, check=True)
            outputs.append(json.loads(process.stdout))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], dump_order(self.root))

    def test_real_nifti_output_and_mask_labels_are_preserved(self):
        ds = build_dataset(self.root, return_full_res_mask=True)
        paths = ds.image_pairs[0]
        image = np.arange(64, dtype=np.float32).reshape(8, 8)
        organ = np.zeros((8, 8), dtype=np.int16)
        organ[:4, :4], organ[:4, 4:] = 1, 6
        organ[4:, :4], organ[4:, 4:] = 9, 12
        lesion = np.zeros((8, 8), dtype=np.int16)
        lesion[4:, 4:] = 7
        body = np.ones((8, 8), dtype=np.int16)
        body[4:, :] = 2
        for path, array in zip(paths, (image, image * 2 + 3, organ, lesion, body)):
            sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
        a, b, small_mask, full_mask, a_path, b_path = ds[0]
        self.assertEqual(a.shape, (1, 8, 8))
        self.assertEqual(b.shape, (1, 8, 8))
        self.assertEqual(small_mask.shape, (3, 2, 2))
        self.assertEqual(full_mask.shape, (3, 8, 8))
        self.assertEqual(a.dtype, torch.float32)
        self.assertEqual(b.dtype, torch.float32)
        self.assertEqual(small_mask.dtype, torch.int16)
        self.assertEqual(full_mask.dtype, torch.int16)
        self.assertEqual(a.device.type, 'cpu')
        self.assertTrue(torch.isfinite(a).all().item())
        self.assertTrue(torch.isfinite(b).all().item())
        self.assertAlmostEqual(a.min().item(), -1.0)
        self.assertAlmostEqual(a.max().item(), 1.0)
        torch.testing.assert_close(a, b)
        np.testing.assert_array_equal(full_mask.numpy(), np.stack((organ, lesion, body)))
        np.testing.assert_array_equal(small_mask.numpy(),
                                      np.array([[[1, 6], [9, 12]],
                                                [[0, 0], [0, 7]],
                                                [[1, 1], [2, 2]]], dtype=np.int16))
        self.assertEqual((a_path, b_path), (paths[0], paths[1]))
        self.assertEqual(len(build_dataset(self.root)[0]), 5)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--dump-order':
        print(json.dumps(dump_order(Path(sys.argv[2])), sort_keys=True))
    else:
        unittest.main()
