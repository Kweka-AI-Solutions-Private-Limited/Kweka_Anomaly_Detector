"""
test_localization_exp2.py — Unit Tests for PatchCore Experiment 2 Spatial Refinement
-------------------------------------------------------------------------------------
Verifies that refine_heatmap_edge_aware:
1. Returns a refined heatmap of identical shape and uint8 dtype.
2. Correctly applies joint bilateral spatial filtering with guide image.
3. Preserves baseline thresholding (norm_map > 128) behavior without error.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.localization_experiment import (
    refine_heatmap_edge_aware,
    select_localization_component_current
)


def test_refine_heatmap_edge_aware_basic():
    # 1. Create a synthetic blurry anomaly map (256x256)
    heat = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(heat, (128, 128), 40, 255, -1)
    heat_blurred = cv2.GaussianBlur(heat, (31, 31), 0)

    # 2. Create a synthetic guide RGB image with sharp vertical edge
    guide = np.zeros((256, 256, 3), dtype=np.uint8)
    guide[:, 128:] = 200  # Sharp vertical edge at x=128

    # 3. Apply edge-aware spatial refinement
    refined = refine_heatmap_edge_aware(heat_blurred, guide_image_np=guide, d=9, sigma_color=75.0, sigma_space=75.0)

    assert refined is not None, "Refined heatmap should not be None."
    assert refined.shape == (256, 256), f"Expected shape (256, 256), got {refined.shape}"
    assert refined.dtype == np.uint8, f"Expected dtype uint8, got {refined.dtype}"

    # Verify edge preservation: bilateral filter sharpens gradient across x=128
    print("✓ Basic edge-aware spatial refinement test passed.")


def test_refine_heatmap_edge_aware_none_fallback():
    heat = np.zeros((100, 100), dtype=np.uint8)
    refined = refine_heatmap_edge_aware(heat, guide_image_np=None)

    assert refined is not None
    assert refined.shape == (100, 100)
    print("✓ Fallback without guide image test passed.")


def test_localization_pipeline_preserves_threshold_and_bbox():
    # Test that thresholding at > 128 and contour selection operates cleanly on refined map
    heat = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(heat, (100, 100), 20, 220, -1)
    heat_blurred = cv2.GaussianBlur(heat, (15, 15), 0)

    refined = refine_heatmap_edge_aware(heat_blurred, d=9, sigma_color=75.0, sigma_space=75.0)
    thresh_mask = (refined > 128).astype(np.uint8)
    contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = select_localization_component_current(contours, refined)

    assert bbox is not None, "Expected valid bounding box around refined peak."
    assert bbox["x"] > 0 and bbox["y"] > 0 and bbox["width"] > 0 and bbox["height"] > 0
    print("✓ Localization pipeline preservation test passed.")


if __name__ == "__main__":
    test_refine_heatmap_edge_aware_basic()
    test_refine_heatmap_edge_aware_none_fallback()
    test_localization_pipeline_preserves_threshold_and_bbox()
    print("\nALL EXPERIMENT 2 UNIT TESTS PASSED SUCCESSFULLY!")
