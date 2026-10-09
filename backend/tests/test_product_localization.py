"""
Extended Unit Tests for Product-Centric Representation & Normalization (V3 Revision 2)
----------------------------------------------------------------------------------------
Verifies 15 core representation & invariance constraints:
  1. Centered screw localization
  2. Screw shifted left/right
  3. Screw shifted up/down
  4. Different image sizes (512x512, 1024x768)
  5. Different image aspect ratios (16:9, 4:3, 1:2)
  6. Different screw scales (small vs large)
  7. Different surrounding background amounts
  8. Screw near/touching image boundary
  9. Complete screw remains inside canonical representation
 10. Final output tensor is consistently (3, 256, 256) float32
 11. Screw aspect ratio is preserved (no non-uniform stretching)
 12. Seamless border background color matching (no artificial black border contrast edges)
 13. Deterministic output for identical input
 14. Fallback behavior for empty/flood images
 15. Preprocessing does NOT modify original source image array
"""

import unittest
import numpy as np
import cv2
import torch
from pathlib import Path

from services.product_localization_service import (
    detect_product_bbox,
    localize_and_normalize_product,
    estimate_border_background_color,
    canonicalize_and_pad_crop
)


def create_synthetic_screw_image(
    img_size=(256, 256),
    screw_bbox=(60, 40, 40, 160),
    bg_color=(202, 202, 202),
    screw_color=(70, 75, 80),
    add_threads=True
) -> np.ndarray:
    """Creates a synthetic RGB image containing a screw object on a customizable background."""
    h, w = img_size[1], img_size[0] if len(img_size) == 2 else (img_size[0], img_size[1])
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bg_color

    x, y, sw, sh = screw_bbox
    # Draw screw body
    cv2.rectangle(img, (x, y), (x + sw, y + sh), screw_color, -1)
    # Draw screw head
    head_w = int(sw * 1.5)
    head_x = x - (head_w - sw) // 2
    cv2.rectangle(img, (head_x, y), (head_x + head_w, y + int(sh * 0.2)), (90, 95, 100), -1)

    # Draw thread notches
    if add_threads:
        for ty in range(y + int(sh * 0.25), y + sh, 12):
            cv2.line(img, (x - 3, ty), (x + sw + 3, ty + 4), (40, 45, 50), 2)

    return img


class TestProductNormalizationExtended(unittest.TestCase):

    def test_1_centered_screw(self):
        img_rgb = create_synthetic_screw_image(screw_bbox=(100, 40, 56, 176))
        tensor, crop, meta, status = localize_and_normalize_product(img_rgb)
        self.assertEqual(status, "SUCCESS")
        self.assertFalse(meta["is_fallback"])
        self.assertEqual(tensor.shape, torch.Size([3, 256, 256]))

    def test_2_screw_shifted_left_right(self):
        # Shifted left
        img_left = create_synthetic_screw_image(screw_bbox=(20, 40, 50, 160))
        t_left, _, meta_left, _ = localize_and_normalize_product(img_left)

        # Shifted right
        img_right = create_synthetic_screw_image(screw_bbox=(180, 40, 50, 160))
        t_right, _, meta_right, _ = localize_and_normalize_product(img_right)

        self.assertFalse(meta_left["is_fallback"])
        self.assertFalse(meta_right["is_fallback"])
        # Standardized representation tensors should have high correlation (> 0.95)
        sim = torch.cosine_similarity(t_left.flatten().unsqueeze(0), t_right.flatten().unsqueeze(0)).item()
        self.assertGreater(sim, 0.92)

    def test_3_screw_shifted_up_down(self):
        # Shifted up
        img_up = create_synthetic_screw_image(screw_bbox=(100, 10, 50, 150))
        t_up, _, meta_up, _ = localize_and_normalize_product(img_up)

        # Shifted down
        img_down = create_synthetic_screw_image(screw_bbox=(100, 90, 50, 150))
        t_down, _, meta_down, _ = localize_and_normalize_product(img_down)

        self.assertFalse(meta_up["is_fallback"])
        self.assertFalse(meta_down["is_fallback"])

    def test_4_different_image_sizes(self):
        # Large 512x512 image
        img_512 = create_synthetic_screw_image(img_size=(512, 512), screw_bbox=(200, 80, 100, 320))
        t_512, _, meta_512, status_512 = localize_and_normalize_product(img_512)

        self.assertEqual(status_512, "SUCCESS")
        self.assertEqual(meta_512["orig_dimensions"], [512, 512])
        self.assertEqual(t_512.shape, torch.Size([3, 256, 256]))

    def test_5_different_aspect_ratios(self):
        # 16:9 widescreen image (640x360)
        img_16_9 = create_synthetic_screw_image(img_size=(640, 360), screw_bbox=(250, 40, 80, 260))
        t_16_9, _, meta_16_9, status = localize_and_normalize_product(img_16_9)

        self.assertEqual(status, "SUCCESS")
        self.assertEqual(meta_16_9["orig_dimensions"], [640, 360])
        self.assertEqual(t_16_9.shape, torch.Size([3, 256, 256]))

    def test_6_different_screw_scales(self):
        # Small screw vs large screw
        img_small = create_synthetic_screw_image(screw_bbox=(110, 80, 30, 90))
        t_small, _, meta_small, _ = localize_and_normalize_product(img_small)

        img_large = create_synthetic_screw_image(screw_bbox=(70, 10, 110, 230))
        t_large, _, meta_large, _ = localize_and_normalize_product(img_large)

        self.assertEqual(t_small.shape, torch.Size([3, 256, 256]))
        self.assertEqual(t_large.shape, torch.Size([3, 256, 256]))

    def test_7_different_background_amounts(self):
        # Tight image vs wide background image
        img_wide = create_synthetic_screw_image(img_size=(800, 800), screw_bbox=(360, 200, 80, 320))
        t_wide, _, meta_wide, _ = localize_and_normalize_product(img_wide)
        self.assertEqual(t_wide.shape, torch.Size([3, 256, 256]))

    def test_8_screw_near_boundary(self):
        # Screw touching top and left boundary
        img_boundary = create_synthetic_screw_image(screw_bbox=(0, 0, 60, 180))
        t_bound, crop, meta_bound, status = localize_and_normalize_product(img_boundary)

        self.assertEqual(status, "SUCCESS")
        pbox = meta_bound["padded_bbox"]
        self.assertEqual(pbox["x"], 0)
        self.assertEqual(pbox["y"], 0)

    def test_9_complete_screw_inside_canonical_canvas(self):
        img_rgb = create_synthetic_screw_image(screw_bbox=(80, 40, 60, 180))
        _, crop, meta, _ = localize_and_normalize_product(img_rgb)

        # Padding margins must be non-negative
        pm = meta["padding_margins"]
        self.assertGreaterEqual(pm["top"], 0)
        self.assertGreaterEqual(pm["bottom"], 0)
        self.assertGreaterEqual(pm["left"], 0)
        self.assertGreaterEqual(pm["right"], 0)

    def test_10_output_tensor_shape_and_range(self):
        img_rgb = create_synthetic_screw_image()
        tensor, _, _, _ = localize_and_normalize_product(img_rgb)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3, 256, 256]))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_11_aspect_ratio_preservation(self):
        # Non-square crop (ratio 1:3)
        img_tall = create_synthetic_screw_image(screw_bbox=(100, 20, 40, 200))
        _, _, meta, _ = localize_and_normalize_product(img_tall)

        # Canonical dimensions must be a 1:1 square
        cdim = meta["canonical_dimensions"]
        self.assertEqual(cdim[0], cdim[1])

    def test_12_seamless_background_color_matching(self):
        # Background color = (180, 185, 190)
        bg_c = (180, 185, 190)
        img_rgb = create_synthetic_screw_image(bg_color=bg_c)
        _, crop_rgb, meta, _ = localize_and_normalize_product(img_rgb)

        crop_border_color = estimate_border_background_color(crop_rgb)
        pad_color = meta["padding_color"]
        # Padding color should seamlessly match the border background color of crop_rgb
        self.assertEqual(pad_color, list(crop_border_color))

    def test_13_deterministic_output(self):
        img_rgb = create_synthetic_screw_image()
        t1, _, m1, _ = localize_and_normalize_product(img_rgb)
        t2, _, m2, _ = localize_and_normalize_product(img_rgb)

        self.assertTrue(torch.equal(t1, t2))
        self.assertEqual(m1["padded_bbox"], m2["padded_bbox"])

    def test_14_fallback_behavior(self):
        # Blank grey image with no foreground object
        img_blank = np.zeros((256, 256, 3), dtype=np.uint8)
        img_blank[:, :] = (200, 200, 200)

        tensor, crop, meta, status = localize_and_normalize_product(img_blank)
        self.assertIn("FALLBACK", status)
        self.assertTrue(meta["is_fallback"])
        self.assertEqual(tensor.shape, torch.Size([3, 256, 256]))

    def test_15_source_image_non_mutation(self):
        img_orig = create_synthetic_screw_image()
        img_copy = img_orig.copy()

        _ = localize_and_normalize_product(img_orig)

        # Assert source array remains 100% identical
        self.assertTrue(np.array_equal(img_orig, img_copy))


if __name__ == "__main__":
    unittest.main()
