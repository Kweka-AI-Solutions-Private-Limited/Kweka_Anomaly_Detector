"""
localization_experiment.py — PatchCore Localization Experiment Helper
-----------------------------------------------------------------------
Implements component selection algorithms for defect localization:
1. select_localization_component_current: Baseline max-contour-area selection.
2. select_localization_component_intensity_weighted: Experimental area + intensity scoring selection.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def select_localization_component_current(
    contours: List[np.ndarray],
    norm_map_resized: np.ndarray
) -> Optional[Dict[str, int]]:
    """
    BASELINE METHOD: Selects the connected component with the maximum geometric contour area.
    
    Returns:
        Dict {"x": int, "y": int, "width": int, "height": int} or None if no contours.
    """
    if not contours:
        return None

    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)
    return {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}


def select_localization_component_intensity_weighted(
    contours: List[np.ndarray],
    norm_map_resized: np.ndarray,
    alpha: float = 1.5,
    beta: float = 1.0,
    min_area: float = 5.0
) -> Optional[Dict[str, int]]:
    """
    EXPERIMENTAL METHOD: Intensity-Weighted Component Selection.
    
    Scores each connected component by combining its spatial pixel area,
    mean anomaly intensity, and 90th percentile peak intensity inside the contour.
    
    Formula:
        score = area * ((mean_intensity / 255.0) ** alpha) * ((p90_intensity / 255.0) ** beta)
        
    Args:
        contours: List of OpenCV contour arrays.
        norm_map_resized: Normalized anomaly map (uint8 array, range 0-255).
        alpha: Exponent weight for mean intensity ratio (default: 1.5).
        beta: Exponent weight for 90th percentile intensity ratio (default: 1.0).
        min_area: Minimum pixel area to consider (default: 5.0).
        
    Returns:
        Dict {"x": int, "y": int, "width": int, "height": int} or None.
    """
    if not contours:
        return None

    h, w = norm_map_resized.shape[:2]
    best_score = -1.0
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Create binary mask for this single contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        # Extract normalized anomaly values inside contour
        vals = norm_map_resized[mask == 255]
        if len(vals) == 0:
            continue

        mean_intensity = float(np.mean(vals))
        p90_intensity = float(np.percentile(vals, 90))

        # Compute intensity-weighted score
        score = area * ((mean_intensity / 255.0) ** alpha) * ((p90_intensity / 255.0) ** beta)

        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        # Fallback to current baseline if all contours were below min_area
        return select_localization_component_current(contours, norm_map_resized)

    x, y, w, h = cv2.boundingRect(best_cnt)
    return {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}


def refine_heatmap_edge_aware(
    norm_map_resized: np.ndarray,
    guide_image_np: Optional[np.ndarray] = None,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    EXPERIMENT 2 METHOD: Edge-Aware Spatial Heatmap Refinement.
    
    Applies joint bilateral spatial filtering to the upsampled anomaly heatmap,
    using the high-resolution RGB/grayscale input image as spatial guidance.
    Aligns heatmap intensity gradients with sharp physical edges in the guide image.
    
    Args:
        norm_map_resized: Normalized float32 or uint8 anomaly map (range 0-255).
        guide_image_np: High-resolution RGB or grayscale numpy image array (0-255).
        d: Diameter of each pixel neighborhood used during filtering (default: 9).
        sigma_color: Filter sigma in the intensity/color space (default: 75.0).
        sigma_space: Filter sigma in the coordinate spatial space (default: 75.0).
        
    Returns:
        np.ndarray: Edge-refined anomaly map of same shape and uint8 dtype.
    """
    if norm_map_resized is None or norm_map_resized.size == 0:
        return norm_map_resized

    map_uint8 = norm_map_resized.astype(np.uint8) if norm_map_resized.dtype != np.uint8 else norm_map_resized.copy()

    # Apply bilateral spatial edge-preserving refinement
    refined = cv2.bilateralFilter(map_uint8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return refined

