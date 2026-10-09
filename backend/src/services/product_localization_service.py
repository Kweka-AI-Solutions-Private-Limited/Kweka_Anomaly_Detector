"""
InspectAI Product Localization & Normalization Service (V3 Revision 2)
------------------------------------------------------------------------
Provides robust, deterministic product-centric representation for Pipeline A Version #3 (R2):
  - Border background sampling & adaptive color/intensity distance
  - Gradient energy / edge saliency integration
  - Morphological closing & contour component merging
  - 6% relative safety margin padding (preserves screw tips, heads, threads)
  - Seamless background-color canvas padding (eliminates artificial border edge artifacts for PatchCore)
  - Conservative, deterministic background luminance stabilization (preserves color/rust/discoloration defects)
  - Aspect-preserving letterbox padding to 1:1 square canvas (256x256)
  - Comprehensive transformation metadata for visual debugging
  - Graceful fallback for invalid/failed localization
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import cv2
import torch
from PIL import Image

DEFAULT_V3_TARGET_SIZE = (256, 256)
DEFAULT_V3_CANONICAL_OCCUPANCY = 0.88
MIN_PRODUCT_AREA_RATIO = 0.02  # 2% of total image area
REVISION_ID = "v3_product_centric_r2"


def estimate_border_background_color(img_rgb: np.ndarray, border_thickness: int = 5) -> Tuple[int, int, int]:
    """
    Estimates the dominant background RGB color by sampling pixels from the outer border of the image/crop.
    """
    h, w = img_rgb.shape[:2]
    bt = min(border_thickness, max(1, h // 10), max(1, w // 10))

    border_pixels = []
    border_pixels.append(img_rgb[:bt, :, :].reshape(-1, 3))
    border_pixels.append(img_rgb[-bt:, :, :].reshape(-1, 3))
    border_pixels.append(img_rgb[:, :bt, :].reshape(-1, 3))
    border_pixels.append(img_rgb[:, -bt:, :].reshape(-1, 3))

    all_border = np.vstack(border_pixels)
    bg_median = np.median(all_border, axis=0)
    return int(round(bg_median[0])), int(round(bg_median[1])), int(round(bg_median[2]))


def detect_product_bbox(
    img_rgb: np.ndarray,
    safety_margin_ratio: float = 0.06
) -> Dict[str, Any]:
    """
    Detects the product bounding box using robust OpenCV color distance + gradient edge saliency.
    """
    h, w = img_rgb.shape[:2]
    total_area = float(h * w)

    if h < 20 or w < 20:
        return {
            "is_valid": False,
            "bbox": {"x": 0, "y": 0, "width": w, "height": h},
            "padded_bbox": {"x": 0, "y": 0, "width": w, "height": h},
            "confidence": 0.0,
            "reason": "IMAGE_TOO_SMALL"
        }

    # 1. Color distance from border background
    bg_rgb = estimate_border_background_color(img_rgb)
    color_diff = np.abs(img_rgb.astype(np.float32) - np.array(bg_rgb, dtype=np.float32))
    color_dist = np.sqrt(np.sum(color_diff ** 2, axis=2))
    color_dist_norm = cv2.normalize(color_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Gradient / Edge saliency
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Combine color distance and gradient energy into a unified saliency map
    saliency = cv2.addWeighted(color_dist_norm, 0.6, grad_mag_norm, 0.4, 0)

    # 4. Otsu thresholding to create binary candidate mask
    blur = cv2.GaussianBlur(saliency, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Morphological closing to bridge thread gaps & fill internal screw body
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "is_valid": False,
            "bbox": {"x": 0, "y": 0, "width": w, "height": h},
            "padded_bbox": {"x": 0, "y": 0, "width": w, "height": h},
            "confidence": 0.0,
            "reason": "NO_CONTOURS_DETECTED"
        }

    # Filter out tiny noise contours (< 0.2% of image area)
    min_cnt_area = total_area * 0.002
    valid_cnts = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_cnt_area]
    if not valid_cnts:
        valid_cnts = contours

    # Select largest component or merge close fragments
    main_cnt = max(valid_cnts, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(main_cnt)

    # If secondary significant contours exist (> 20% of main contour area), merge bounding boxes
    main_area = cv2.contourArea(main_cnt)
    for cnt in valid_cnts:
        if cv2.contourArea(cnt) > (main_area * 0.2):
            x, y, bw, bh = cv2.boundingRect(cnt)
            min_x = min(rx, x)
            min_y = min(ry, y)
            max_x = max(rx + rw, x + bw)
            max_y = max(ry + rh, y + bh)
            rx, ry, rw, rh = min_x, min_y, max_x - min_x, max_y - min_y

    raw_area = float(rw * rh)
    raw_ratio = raw_area / total_area

    # Validation Guard #1: Product region too small
    if raw_ratio < MIN_PRODUCT_AREA_RATIO:
        return {
            "is_valid": False,
            "bbox": {"x": int(rx), "y": int(ry), "width": int(rw), "height": int(rh)},
            "padded_bbox": {"x": 0, "y": 0, "width": w, "height": h},
            "confidence": round(raw_ratio, 4),
            "reason": f"DETECTED_AREA_TOO_SMALL ({raw_ratio*100:.1f}% < {MIN_PRODUCT_AREA_RATIO*100:.1f}%)"
        }

    # Validation Guard #2: Mask covers almost entire image (> 98%) indicating background flood
    if raw_ratio > 0.98:
        return {
            "is_valid": False,
            "bbox": {"x": 0, "y": 0, "width": w, "height": h},
            "padded_bbox": {"x": 0, "y": 0, "width": w, "height": h},
            "confidence": round(raw_ratio, 4),
            "reason": "BACKGROUND_FLOOD_DETECTED"
        }

    # Apply relative safety padding (5-8%) to avoid clipping screw tips/threads/heads
    pad_w = int(round(rw * safety_margin_ratio))
    pad_h = int(round(rh * safety_margin_ratio))

    padded_x = max(0, rx - pad_w)
    padded_y = max(0, ry - pad_h)
    padded_w = min(w - padded_x, rw + (2 * pad_w))
    padded_h = min(h - padded_y, rh + (2 * pad_h))

    return {
        "is_valid": True,
        "bbox": {"x": int(rx), "y": int(ry), "width": int(rw), "height": int(rh)},
        "padded_bbox": {"x": int(padded_x), "y": int(padded_y), "width": int(padded_w), "height": int(padded_h)},
        "confidence": round(min(1.0, raw_ratio * 3.0), 4),
        "reason": "LOCALIZATION_SUCCESS"
    }


def canonicalize_and_pad_crop(
    crop_rgb: np.ndarray,
    target_occupancy: float = DEFAULT_V3_CANONICAL_OCCUPANCY,
    pad_color: Optional[Tuple[int, int, int]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pads localized crop symmetrically into a 1:1 square canvas matching canonical occupancy (default 88%).
    Uses border background color matching to ensure zero artificial contrast box edges around the crop.
    """
    ch, cw = crop_rgb.shape[:2]
    max_dim = max(ch, cw)

    if pad_color is None:
        pad_color = estimate_border_background_color(crop_rgb)

    canvas_dim = max(1, int(round(max_dim / float(target_occupancy))))

    pad_top = (canvas_dim - ch) // 2
    pad_bottom = canvas_dim - ch - pad_top
    pad_left = (canvas_dim - cw) // 2
    pad_right = canvas_dim - cw - pad_left

    canonical = cv2.copyMakeBorder(
        crop_rgb,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )

    padding_info = {
        "canonical_dim": canvas_dim,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_color": list(pad_color)
    }

    return canonical, padding_info


def apply_conservative_luminance_stabilization(
    img_rgb: np.ndarray,
    target_bg_lum: float = 200.0,
    max_gain_adjustment: float = 0.25
) -> np.ndarray:
    """
    Applies conservative, linear luminance scaling relative to background median.
    Reduces moderate lighting variations across environments while strictly preserving color/rust/discoloration defects.
    """
    bg_rgb = estimate_border_background_color(img_rgb)
    bg_lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]

    if bg_lum < 10.0:
        return img_rgb  # Avoid divide-by-zero on black images

    gain = target_bg_lum / bg_lum
    # Restrict gain adjustment to conservative bounds [1 - max_gain, 1 + max_gain]
    gain_clamped = float(np.clip(gain, 1.0 - max_gain_adjustment, 1.0 + max_gain_adjustment))

    if abs(gain_clamped - 1.0) < 0.02:
        return img_rgb

    scaled = img_rgb.astype(np.float32) * gain_clamped
    scaled_clipped = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return scaled_clipped


def localize_and_normalize_product(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = DEFAULT_V3_TARGET_SIZE,
    safety_margin_ratio: float = 0.06,
    target_occupancy: float = DEFAULT_V3_CANONICAL_OCCUPANCY,
    enable_luminance_stabilization: bool = True
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any], str]:
    """
    Main product-centric preprocessing entrypoint (V3 Revision 2).
    Loads image, detects product, crops with safety margin, canonicalizes framing, applies background-matched padding,
    converts to (3, 256, 256) float tensor, and compiles visual debugging metadata.
    
    Returns:
    (processed_tensor, localized_crop_rgb, transformation_metadata, localization_status)
    """
    # 1. Load image as RGB numpy array (ensure source copy to prevent input mutation)
    if isinstance(image_input, (str, Path)):
        img_pil = Image.open(Path(image_input)).convert("RGB")
        img_rgb = np.array(img_pil)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img_rgb = cv2.cvtColor(image_input.copy(), cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 3:
            # Assume BGR if coming from cv2.imread, unless already RGB
            img_rgb = cv2.cvtColor(image_input.copy(), cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input.copy()
    else:
        raise TypeError(f"Unsupported image_input type '{type(image_input)}'.")

    orig_h, orig_w = img_rgb.shape[:2]

    # Optional conservative luminance stabilization
    if enable_luminance_stabilization:
        img_rgb = apply_conservative_luminance_stabilization(img_rgb)

    # 2. Detect product bounding box
    det_res = detect_product_bbox(img_rgb, safety_margin_ratio=safety_margin_ratio)

    if det_res["is_valid"]:
        pbox = det_res["padded_bbox"]
        x, y, w, h = pbox["x"], pbox["y"], pbox["width"], pbox["height"]
        crop_rgb = img_rgb[y : y + h, x : x + w]
        localization_status = "SUCCESS"
    else:
        # Graceful fallback to centered 90% crop of original image
        cx = int(orig_w * 0.05)
        cy = int(orig_h * 0.05)
        cw = int(orig_w * 0.90)
        ch = int(orig_h * 0.90)
        crop_rgb = img_rgb[cy : cy + ch, cx : cx + cw]
        localization_status = f"FALLBACK ({det_res['reason']})"

    # 3. Canonicalize framing into 1:1 square canvas with border background color matching
    border_bg = estimate_border_background_color(crop_rgb)
    canonical_rgb, pad_info = canonicalize_and_pad_crop(
        crop_rgb,
        target_occupancy=target_occupancy,
        pad_color=border_bg
    )

    # 4. Resize to target (256, 256) preserving aspect ratio via canonical framing
    resample_mode = getattr(Image, "Resampling", Image).BILINEAR if hasattr(getattr(Image, "Resampling", None), "BILINEAR") else getattr(Image, "BILINEAR", 2)
    pil_resized = Image.fromarray(canonical_rgb).resize(target_size, resample_mode)

    # 5. Convert to PyTorch float32 tensor (3, H, W) in range [0, 1]
    arr = np.array(pil_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    # 6. Build comprehensive visual debugging transformation metadata
    transform_metadata = {
        "revision": REVISION_ID,
        "orig_dimensions": [int(orig_w), int(orig_h)],
        "detected_bbox": det_res["bbox"],
        "padded_bbox": det_res["padded_bbox"],
        "crop_dimensions": [int(crop_rgb.shape[1]), int(crop_rgb.shape[0])],
        "canonical_dimensions": [int(pad_info["canonical_dim"]), int(pad_info["canonical_dim"])],
        "padding_margins": {
            "top": int(pad_info["pad_top"]),
            "bottom": int(pad_info["pad_bottom"]),
            "left": int(pad_info["pad_left"]),
            "right": int(pad_info["pad_right"])
        },
        "padding_color": pad_info["pad_color"],
        "final_dimensions": [int(target_size[0]), int(target_size[1])],
        "occupancy_ratio": target_occupancy,
        "is_fallback": not det_res["is_valid"],
        "fallback_reason": det_res.get("reason") if not det_res["is_valid"] else None,
        "localization_confidence": det_res.get("confidence", 0.0)
    }

    return tensor, crop_rgb, transform_metadata, localization_status
