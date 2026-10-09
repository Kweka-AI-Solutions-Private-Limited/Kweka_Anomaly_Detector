"""
InspectAI Freeform Contour Service
-----------------------------------
Extracts 0-1000 scaled freeform polygon contours for product instances and defects
in Pipeline B Multi-Instance inspection visualizer.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


def extract_product_contour(
    img_bgr: np.ndarray,
    bbox: Dict[str, Any]
) -> List[List[int]]:
    """
    Extracts a smooth product boundary polygon [y, x] scaled 0..1000 for SVG rendering.
    """
    h, w = img_bgr.shape[:2]
    x, y, bw, bh = int(bbox.get("x", 0)), int(bbox.get("y", 0)), int(bbox.get("width", w)), int(bbox.get("height", h))
    
    # Ensure crop inside bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    
    if x2 <= x1 or y2 <= y1:
        # Fallback to rectangle points
        return [
            [int((y1/h)*1000), int((x1/w)*1000)],
            [int((y1/h)*1000), int((x2/w)*1000)],
            [int((y2/h)*1000), int((x2/w)*1000)],
            [int((y2/h)*1000), int((x1/w)*1000)]
        ]
    
    crop = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Thresholding & Morphological smoothing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [
            [int((y1/h)*1000), int((x1/w)*1000)],
            [int((y1/h)*1000), int((x2/w)*1000)],
            [int((y2/h)*1000), int((x2/w)*1000)],
            [int((y2/h)*1000), int((x1/w)*1000)]
        ]
    
    c = max(contours, key=cv2.contourArea)
    epsilon = 0.015 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    
    poly_1000 = []
    for pt in approx:
        px = x1 + int(pt[0][0])
        py = y1 + int(pt[0][1])
        scaled_y = int((py / h) * 1000)
        scaled_x = int((px / w) * 1000)
        poly_1000.append([scaled_y, scaled_x])
        
    return poly_1000


def extract_defect_contour(
    img_bgr: np.ndarray,
    bbox: Dict[str, Any]
) -> List[List[int]]:
    """
    Extracts defect localized polygon [y, x] scaled 0..1000.
    """
    return extract_product_contour(img_bgr, bbox)
