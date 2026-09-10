"""CPU regression coverage for the public model and memory interfaces.

Run from the repository root:
    python -m unittest discover -s tests -p test_public_model_interfaces.py -v

Uses genuine small models, without mocks, downloaded weights or MRI data.
"""
import os
from pathlib import Path
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Some installed xFormers builds probe CUDA devices while importing their
# optional Triton kernels, even though this suite uses only CPU attention.
os.environ['XFORMERS_ENABLE_TRITON'] = '0'
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from Memory_pair_v7 import MemorySharedPartsV7, SEG_IDX_V7
from Models_v2 import AutoencoderKL


class PublicModelInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def build_model(self):
        return AutoencoderKL(
            spatial_dims=2, in_channels=1, out_channels=1,
            num_res_blocks=1, num_channels=[8, 16],
            attention_levels=[False, False], latent_channels=8,
            norm_num_groups=4, with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False, use_flash_attention=False,
            use_convtranspose=True, sty_ch=4,
        ).cpu().eval()

    def check_forward(self, label):
        # Restoring the RNG isolates this suite from other CPU tests. The same
        # sampling seed lets the explicit public decoder calls define the
        # expected reconstruction, translation and return ordering.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(11)
            model = self.build_model()
            image = torch.randn(2, 1, 16, 16)
            torch.manual_seed(29)
            actual_a, actual_b, actual_mu = model(image, label)

            torch.manual_seed(29)
            mu, sigma = model.encode(image)
            source_sample = model.sampling(mu, sigma)
            target_sample = model.sampling(-mu, sigma)
            if label == 'A':
                expected_a = model.decode_A_raw(source_sample)
                expected_b = model.decode_B_raw(target_sample)
            else:
                expected_b = model.decode_B_raw(source_sample)
                expected_a = model.decode_A_raw(target_sample)

            self.assertEqual(actual_a.shape, image.shape)
            self.assertEqual(actual_b.shape, image.shape)
            self.assertEqual(actual_mu.shape, (2, 8, 8, 8))
            for value in (actual_a, actual_b, actual_mu):
                self.assertEqual(value.device.type, 'cpu')
                self.assertTrue(torch.isfinite(value).all().item())
            torch.testing.assert_close(actual_a, expected_a)
            torch.testing.assert_close(actual_b, expected_b)
            torch.testing.assert_close(actual_mu, mu)

    def test_forward_a_reconstructs_a_and_translates_to_b(self):
        self.check_forward('A')

    def test_forward_b_reconstructs_b_and_translates_to_a(self):
        self.check_forward('B')

    def test_forward_rejects_unknown_domain(self):
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            model = self.build_model()
            with self.assertRaisesRegex(ValueError, "label must be 'A' or 'B'"):
                model(torch.zeros(2, 1, 16, 16), 'unknown')

    def test_memory_constructor_uses_defined_v7_label_order_by_default(self):
        memory = MemorySharedPartsV7(
            memory_size=[2] * len(SEG_IDX_V7), kdim=8, vdim=4,
        )
        self.assertEqual(memory.seg_idx, SEG_IDX_V7)
        self.assertEqual(len(memory.memory_size), len(memory.seg_idx))


if __name__ == '__main__':
    unittest.main()
