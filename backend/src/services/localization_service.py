"""
InspectAI Localization Service
------------------------------
Production defect localization helper functions:
- Intensity-weighted connected component selection for defect bounding boxes
- Baseline contour selection fallback
"""

from typing import Dict, List, Optional
import cv2
import numpy as np


def select_localization_component_current(
    contours: List[np.ndarray],
    norm_map_resized: np.ndarray
) -> Optional[Dict[str, int]]:
    """
    BASELINE METHOD: Selects the connected component with the maximum geometric contour area.
    Returns: Dict {"x": int, "y": int, "width": int, "height": int} or None if no contours.
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
    Intensity-Weighted Component Selection.
    Scores each connected component by combining spatial pixel area,
    mean anomaly intensity, and 90th percentile peak intensity inside the contour.
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

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        vals = norm_map_resized[mask == 255]
        if len(vals) == 0:
            continue

        mean_intensity = float(np.mean(vals))
        p90_intensity = float(np.percentile(vals, 90))

        score = area * ((mean_intensity / 255.0) ** alpha) * ((p90_intensity / 255.0) ** beta)

        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return select_localization_component_current(contours, norm_map_resized)

    x, y, w, h = cv2.boundingRect(best_cnt)
    return {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
