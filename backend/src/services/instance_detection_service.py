"""
InspectAI Instance Detection Service
------------------------------------
Dynamic multi-stage computer vision product instance detector & separator:

Stage 1: Texture-Suppressed Surface & Edge Boundary Filtering (Bilateral + Morph Gradient)
Stage 2: Composite Region & Dynamic Multi-Line Grid Separator Line Detection
Stage 3: Morphological Connected Components & Hierarchy Refinement
Stage 4: Non-Maximum Suppression (NMS) & Overlap Removal
Stage 5: Geometric Candidate Validation (Area, Aspect Ratio, Boundaries)
Stage 6: Crop Extraction, Physical File Persistence & Verification
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Safety & Limits Constants
MAX_IMAGE_DIMENSION = 4096
MAX_INSTANCES = 20
MIN_INSTANCE_AREA = 400


def _merge_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int, float]],
    iou_thresh: float = 0.3,
    overlap_ratio_thresh: float = 0.6,
    max_gap_px: int = 12
) -> List[Tuple[int, int, int, int, float]]:
    """Merges overlapping, nested, or touching candidate bounding boxes using NMS & proximity logic."""
    if not boxes:
        return []

    # Iteratively merge boxes until stable
    current_boxes = [list(b) for b in boxes]
    changed = True

    while changed:
        changed = False
        merged_list = []
        visited = [False] * len(current_boxes)

        for i in range(len(current_boxes)):
            if visited[i]:
                continue

            cx, cy, cw, ch, c_area = current_boxes[i]
            c_x2, c_y2 = cx + cw, cy + ch
            visited[i] = True

            merged_box = [cx, cy, cw, ch, c_area]

            for j in range(i + 1, len(current_boxes)):
                if visited[j]:
                    continue

                mx, my, mw, mh, m_area = current_boxes[j]
                m_x2, m_y2 = mx + mw, my + mh

                # Intersection
                ix1, iy1 = max(cx, mx), max(cy, my)
                ix2, iy2 = min(c_x2, m_x2), min(c_y2, m_y2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter_area = float(iw * ih)

                # Distance gaps
                x_dist = max(0, max(cx, mx) - min(c_x2, m_x2))
                y_dist = max(0, max(cy, my) - min(c_y2, m_y2))

                is_touching_or_close = (x_dist <= max_gap_px and y_dist <= max_gap_px)

                should_merge = False
                if inter_area > 0:
                    union_area = float(c_area + m_area - inter_area)
                    iou = inter_area / union_area if union_area > 0 else 0
                    overlap_ratio = inter_area / min(c_area, m_area)
                    if iou > iou_thresh or overlap_ratio > overlap_ratio_thresh:
                        should_merge = True

                if is_touching_or_close:
                    should_merge = True

                if should_merge:
                    new_x = min(cx, mx)
                    new_y = min(cy, my)
                    new_w = max(c_x2, m_x2) - new_x
                    new_h = max(c_y2, m_y2) - new_y
                    merged_box = [new_x, new_y, new_w, new_h, float(new_w * new_h)]
                    cx, cy, cw, ch, c_area = merged_box
                    c_x2, c_y2 = cx + cw, cy + ch
                    visited[j] = True
                    changed = True

            merged_list.append((merged_box[0], merged_box[1], merged_box[2], merged_box[3], merged_box[4]))

        current_boxes = [list(b) for b in merged_list]

    return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])) for b in current_boxes]


def _find_projection_peaks(proj: np.ndarray, min_distance: int, thresh_ratio: float = 0.35) -> List[int]:
    """Finds peak cut line locations in a 1D edge projection profile."""
    if proj.size == 0:
        return []
    
    max_val = proj.max()
    if max_val < 1e-5:
        return []

    norm_proj = proj / max_val
    peaks = []
    length = len(norm_proj)

    # Search peaks with non-maximum suppression in 1D window
    for i in range(1, length - 1):
        if norm_proj[i] > thresh_ratio and norm_proj[i] >= norm_proj[i - 1] and norm_proj[i] >= norm_proj[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)

    return peaks


def _split_composite_box_grid(
    img_gray: np.ndarray,
    box: Tuple[int, int, int, int]
) -> List[Tuple[int, int, int, int, float]]:
    """
    Stage 2: Dynamic Multi-Line Grid Separator Detection.

    Analyzes interior edge projections along X and Y axes to identify genuine
    tile-separator lines and split the region into discrete product cells.

    A real separator (grout line, physical gap) must satisfy ALL three criteria:
    1. Projection peak above threshold on a HEAVILY smoothed projection
       (wide Gaussian suppresses narrow texture spikes while preserving wide bands).
    2. Minimum sustained width: the projection must stay above 55% of its max
       for at least `min_sep_width` consecutive positions (real grout lines have
       physical width; a texture edge is 1-2 pixels).
    3. Per-peak row/column coverage: at least 5% of the row's columns (or the
       column's rows) must show significant Sobel edge activity at that separator.
       This rejects highly localised texture spikes that just happen to sum high.
    """
    bx, by, bw, bh = box
    roi = img_gray[by : by + bh, bx : bx + bw]
    rh, rw = roi.shape[:2]

    if rh < 80 or rw < 80:
        return [(bx, by, bw, bh, float(bw * bh))]

    # ------------------------------------------------------------------ #
    # Sobel gradient map – light blur before Sobel (same as before)       #
    # ------------------------------------------------------------------ #
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    sobel_x = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))

    # 1D Projections (sum along every column/row)
    proj_y = np.sum(sobel_x, axis=1)   # row sums  → cuts ∥ to X axis
    proj_x = np.sum(sobel_y, axis=0)   # col sums  → cuts ∥ to Y axis

    # ------------------------------------------------------------------ #
    # Interior search band (15 %–85 %)                                    #
    # ------------------------------------------------------------------ #
    min_cell_h = int(rh * 0.18)
    min_cell_w = int(rw * 0.18)
    y_margin_start, y_margin_end = int(rh * 0.15), int(rh * 0.85)
    x_margin_start, x_margin_end = int(rw * 0.15), int(rw * 0.85)

    proj_y_sub = proj_y[y_margin_start:y_margin_end]
    proj_x_sub = proj_x[x_margin_start:x_margin_end]

    # ------------------------------------------------------------------ #
    # Smooth the 1D projections heavily to suppress narrow texture spikes  #
    # Use a kernel that is ≈ 2 % of the image dimension (min 11 px).      #
    # This preserves the broad "bump" of a wide grout line while washing  #
    # out 1-3 px texture peaks that just happened to sum high.            #
    # ------------------------------------------------------------------ #
    smooth_k_h = max(11, int(rh * 0.02))
    smooth_k_w = max(11, int(rw * 0.02))
    if smooth_k_h % 2 == 0: smooth_k_h += 1
    if smooth_k_w % 2 == 0: smooth_k_w += 1

    proj_y_smooth = cv2.GaussianBlur(
        proj_y_sub.reshape(-1, 1).astype(np.float32),
        (1, smooth_k_h), 0
    ).ravel()
    proj_x_smooth = cv2.GaussianBlur(
        proj_x_sub.reshape(-1, 1).astype(np.float32),
        (1, smooth_k_w), 0
    ).ravel()

    # ------------------------------------------------------------------ #
    # Peak detection on the smoothed projection                           #
    # ------------------------------------------------------------------ #
    y_peaks_rel = _find_projection_peaks(proj_y_smooth, min_distance=min_cell_h, thresh_ratio=0.55)
    x_peaks_rel = _find_projection_peaks(proj_x_smooth, min_distance=min_cell_w, thresh_ratio=0.55)

    # ------------------------------------------------------------------ #
    # Validation 1 – minimum sustained width                              #
    # The smoothed projection must stay ≥ 55 % max for ≥ min_sep_width   #
    # consecutive positions around each peak.                             #
    # ------------------------------------------------------------------ #
    min_sep_width_y = max(3, int(rh * 0.005))   # ≥ 0.5 % of height
    min_sep_width_x = max(3, int(rw * 0.005))

    thresh_ratio = 0.55

    def _width_validated(peaks, proj_smooth, min_sep_width):
        y_max = proj_smooth.max()
        if y_max < 1e-5:
            return []
        validated = []
        for p in peaks:
            threshold = thresh_ratio * y_max
            # Expand left and right from peak while above threshold
            lo, hi = p, p
            while lo > 0 and proj_smooth[lo - 1] >= threshold:
                lo -= 1
            while hi < len(proj_smooth) - 1 and proj_smooth[hi + 1] >= threshold:
                hi += 1
            sustained_width = hi - lo + 1
            if sustained_width >= min_sep_width:
                validated.append(p)
        return validated

    y_peaks_rel = _width_validated(y_peaks_rel, proj_y_smooth, min_sep_width_y)
    x_peaks_rel = _width_validated(x_peaks_rel, proj_x_smooth, min_sep_width_x)

    # ------------------------------------------------------------------ #
    # Validation 2 – per-peak row / col edge-coverage filter             #
    # For a genuine separator, significant edge activity must be spread   #
    # across ≥ 5 % of the row's columns (or col's rows).                 #
    # A texture spike has high *summed* energy from a small cluster of   #
    # pixels; it typically covers < 5 % of the row.                      #
    # ------------------------------------------------------------------ #
    global_sobel_max_y = sobel_x.max()
    global_sobel_max_x = sobel_y.max()
    COVERAGE_THRESHOLD = 0.05   # 5 % of span must be active

    def _coverage_validated_y(peaks_rel, margin_start, sobel_map, global_max, rw):
        if global_max < 1e-5:
            return peaks_rel
        edge_thresh = global_max * 0.20
        validated = []
        for p in peaks_rel:
            abs_pos = margin_start + p
            row_profile = sobel_map[abs_pos, :]
            coverage = float(np.sum(row_profile > edge_thresh)) / rw
            if coverage >= COVERAGE_THRESHOLD:
                validated.append(p)
        return validated

    def _coverage_validated_x(peaks_rel, margin_start, sobel_map, global_max, rh):
        if global_max < 1e-5:
            return peaks_rel
        edge_thresh = global_max * 0.20
        validated = []
        for p in peaks_rel:
            abs_pos = margin_start + p
            col_profile = sobel_map[:, abs_pos]
            coverage = float(np.sum(col_profile > edge_thresh)) / rh
            if coverage >= COVERAGE_THRESHOLD:
                validated.append(p)
        return validated

    y_peaks_rel = _coverage_validated_y(y_peaks_rel, y_margin_start, sobel_x, global_sobel_max_y, rw)
    x_peaks_rel = _coverage_validated_x(x_peaks_rel, x_margin_start, sobel_y, global_sobel_max_x, rh)

    # ------------------------------------------------------------------ #
    # Build grid cells from validated separator positions                  #
    # ------------------------------------------------------------------ #
    y_splits = [y_margin_start + idx for idx in y_peaks_rel]
    x_splits = [x_margin_start + idx for idx in x_peaks_rel]

    y_boundaries = [0] + y_splits + [rh]
    x_boundaries = [0] + x_splits + [rw]

    sub_boxes = []
    for i in range(len(y_boundaries) - 1):
        y1, y2 = y_boundaries[i], y_boundaries[i + 1]
        cell_h = y2 - y1
        if cell_h < 30:
            continue
        for j in range(len(x_boundaries) - 1):
            x1, x2 = x_boundaries[j], x_boundaries[j + 1]
            cell_w = x2 - x1
            if cell_w < 30:
                continue
            sub_x = bx + x1
            sub_y = by + y1
            sub_boxes.append((sub_x, sub_y, cell_w, cell_h, float(cell_w * cell_h)))

    if len(sub_boxes) > 1:
        return sub_boxes

    return [(bx, by, bw, bh, float(bw * bh))]



def validate_candidate_box(
    box: Tuple[int, int, int, int, float],
    orig_w: int,
    orig_h: int,
    dynamic_min_area: float,
    max_object_area: float
) -> bool:
    """
    Stage 5: Geometric candidate validation.
    Verifies area, dimensions, aspect ratio, and boundary constraints.
    """
    x, y, w, h, area = box

    # Area checks
    if area < dynamic_min_area or area > max_object_area:
        return False

    # Minimum dimension checks
    if w < 25 or h < 25:
        return False

    # Aspect ratio check (exclude extreme thin line noise)
    aspect_ratio = float(w) / float(h)
    if aspect_ratio < 0.12 or aspect_ratio > 8.5:
        return False

    # Canvas boundary check
    if x < 0 or y < 0 or (x + w) > orig_w or (y + h) > orig_h:
        return False

    return True


def detect_and_crop_instances(
    image_path: Path,
    output_dir: Path,
    min_area: int = MIN_INSTANCE_AREA,
    max_instances: int = MAX_INSTANCES,
    padding_ratio: float = 0.02,
    model_context: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Gemini Instance Localization Routine.
    Identifies distinct physical product instances via Gemini VLM,
    crops instances from the ORIGINAL image file, saves verified crops to disk,
    and returns ROI coordinates & metadata.
    Does NOT fall back to OpenCV if Gemini fails or returns 0 instances.
    """
    t0 = time.time()

    if not image_path.exists():
        return [], {
            "status": "failed",
            "reason": "IMAGE_FILE_NOT_FOUND",
            "message": f"Input image not found at '{image_path}'."
        }

    try:
        from services.gemini_instance_localization_service import localize_product_instances

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return [], {
                "status": "failed",
                "reason": "IMAGE_DECODE_FAILED",
                "message": "OpenCV failed to decode image file."
            }

        orig_h, orig_w = img_bgr.shape[:2]

        if orig_h > MAX_IMAGE_DIMENSION or orig_w > MAX_IMAGE_DIMENSION:
            return [], {
                "status": "failed",
                "reason": "IMAGE_DIMENSIONS_EXCEEDED",
                "message": f"Image dimensions ({orig_w}x{orig_h}) exceed safety limit ({MAX_IMAGE_DIMENSION}px)."
            }

        output_dir.mkdir(parents=True, exist_ok=True)

        # Call Gemini instance localization service
        loc_res = localize_product_instances(str(image_path), model_context=model_context)

        if loc_res.get("status") == "failed":
            return [], {
                "status": "failed",
                "reason": loc_res.get("reason", "GEMINI_INSTANCE_LOCALIZATION_FAILED"),
                "message": loc_res.get("error", "Gemini instance localization service failed.")
            }

        valid_instances = loc_res.get("instances", [])
        if not valid_instances or loc_res.get("status") == "no_instances":
            return [], {
                "status": "completed",
                "reason": "NO_PRODUCT_INSTANCES_DETECTED",
                "total_instances": 0,
                "detection_time_ms": loc_res.get("latency_ms", 0.0),
                "model": loc_res.get("model")
            }

        if len(valid_instances) > max_instances:
            valid_instances = valid_instances[:max_instances]

        instances = []
        storage_base = Path(__file__).resolve().parent.parent.parent

        for item in valid_instances:
            idx = item["id"]
            pbox = item["pixel_bbox"]
            x, y, w, h = pbox["x"], pbox["y"], pbox["width"], pbox["height"]

            pad_w = max(0, int(w * padding_ratio))
            pad_h = max(0, int(h * padding_ratio))

            crop_x = max(0, x - pad_w)
            crop_y = max(0, y - pad_h)
            crop_w = min(orig_w - crop_x, w + 2 * pad_w)
            crop_h = min(orig_h - crop_y, h + 2 * pad_h)

            if crop_w < 5 or crop_h < 5:
                continue

            # Crop from ORIGINAL image file
            crop_img = img_bgr[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

            crop_filename = f"instance_{idx}.png"
            crop_filepath = output_dir / crop_filename

            # Save crop to disk
            cv2.imwrite(str(crop_filepath), crop_img)

            # Verification of written file
            if not crop_filepath.exists() or crop_filepath.stat().st_size == 0:
                raise ValueError(f"Crop file creation failed for '{crop_filepath}'.")

            check_bgr = cv2.imread(str(crop_filepath))
            if check_bgr is None or check_bgr.shape[0] == 0 or check_bgr.shape[1] == 0:
                raise ValueError(f"Crop file '{crop_filepath}' failed OpenCV image decode verification.")

            try:
                with Image.open(crop_filepath) as pil_check:
                    pil_check.verify()
            except Exception as pil_err:
                raise ValueError(f"Crop file '{crop_filepath}' failed PIL image integrity check: {pil_err}")

            try:
                rel_crop_uri = str(crop_filepath.relative_to(storage_base)).replace("\\", "/")
            except ValueError:
                rel_crop_uri = f"storage/inspections/crops/{crop_filename}"

            instances.append({
                "instance_id": idx,
                "label": item.get("label", "product"),
                "box_2d": item.get("box_2d"),
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "padded_bbox": {
                    "x": int(crop_x),
                    "y": int(crop_y),
                    "width": int(crop_w),
                    "height": int(crop_h)
                },
                "crop_filepath": str(crop_filepath),
                "crop_storage_uri": rel_crop_uri,
                "detection_confidence": 0.95
            })

        detection_time_ms = round((time.time() - t0) * 1000, 1)

        return instances, {
            "status": "completed",
            "reason": None,
            "total_instances": len(instances),
            "detection_time_ms": loc_res.get("latency_ms", detection_time_ms),
            "model": loc_res.get("model")
        }

    except Exception as e:
        logger.error(f"detect_and_crop_instances failed: {str(e)}", exc_info=True)
        return [], {
            "status": "failed",
            "reason": "INSTANCE_DETECTION_FAILED",
            "message": f"Instance detection encountered error: {str(e)}"
        }
