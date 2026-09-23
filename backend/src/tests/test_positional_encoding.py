"""
test_positional_encoding.py — Unit Tests for Spatial Positional Encoding Experiment
--------------------------------------------------------------------------------------
Verifies:
1. PE matrix shape and dimension requirements (e.g. 32x32 -> (1024, 16)).
2. Determinism of positional encoding generation.
3. Feature concatenation correctness:
   - Same visual feature + same position -> same enhanced feature
   - Same visual feature + different position -> different enhanced feature
4. Train/Inference feature dimension equality (D + pos_dim).
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import unittest
import torch
import numpy as np

from services.positional_encoding_experiment import (
    generate_2d_sinusoidal_positional_encoding,
    augment_patch_features_with_position
)


class TestPositionalEncoding(unittest.TestCase):

    def test_pe_dimensions(self):
        H, W = 32, 32
        pos_dim = 16
        pe = generate_2d_sinusoidal_positional_encoding(H, W, pos_dim=pos_dim)
        self.assertEqual(pe.shape, (H * W, pos_dim))
        self.assertEqual(pe.dtype, torch.float32)

    def test_pe_determinism(self):
        H, W = 32, 32
        pos_dim = 16
        pe1 = generate_2d_sinusoidal_positional_encoding(H, W, pos_dim=pos_dim)
        pe2 = generate_2d_sinusoidal_positional_encoding(H, W, pos_dim=pos_dim)
        torch.testing.assert_close(pe1, pe2)

    def test_feature_augmentation_unbatched(self):
        H, W = 32, 32
        N = H * W
        D = 512
        pos_dim = 16

        visual_features = torch.randn(N, D)
        enhanced = augment_patch_features_with_position(visual_features, H, W, pos_dim=pos_dim)
        self.assertEqual(enhanced.shape, (N, D + pos_dim))

    def test_feature_augmentation_batched(self):
        B = 4
        H, W = 32, 32
        N = H * W
        D = 512
        pos_dim = 16

        visual_features = torch.randn(B, N, D)
        enhanced = augment_patch_features_with_position(visual_features, H, W, pos_dim=pos_dim)
        self.assertEqual(enhanced.shape, (B, N, D + pos_dim))

    def test_same_feature_different_positions(self):
        H, W = 32, 32
        N = H * W
        D = 512
        pos_dim = 16

        # Two identical visual features at different spatial indices (index 0 vs index 100)
        v = torch.randn(1, D)
        v_expanded = v.expand(N, -1)

        enhanced = augment_patch_features_with_position(v_expanded, H, W, pos_dim=pos_dim)

        patch_0 = enhanced[0]     # Top-left corner (y=0, x=0)
        patch_100 = enhanced[100] # Different coordinate (y=3, x=4)

        # Visual parts must be identical
        torch.testing.assert_close(patch_0[:D], patch_100[:D])

        # Enhanced full features must be DIFFERENT due to positional encoding
        self.assertFalse(torch.allclose(patch_0, patch_100))

    def test_train_test_dimension_match(self):
        H, W = 32, 32
        N = H * W
        D = 1024
        pos_dim = 16

        train_feats = torch.randn(N, D)
        test_feats = torch.randn(N, D)

        enhanced_train = augment_patch_features_with_position(train_feats, H, W, pos_dim=pos_dim)
        enhanced_test = augment_patch_features_with_position(test_feats, H, W, pos_dim=pos_dim)

        self.assertEqual(enhanced_train.shape[1], enhanced_test.shape[1])
        self.assertEqual(enhanced_train.shape[1], D + pos_dim)


if __name__ == "__main__":
    unittest.main()
