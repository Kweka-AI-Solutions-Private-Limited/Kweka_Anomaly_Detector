"""
InspectAI Shared Instance Preprocessing Service
------------------------------------------------
Provides aspect-preserving, letterbox-padded preprocessing for product instance crops.
Used identically across:
  1. Pipeline B memory-bank building (GOOD instance crops)
  2. Pipeline B out-of-sample GOOD threshold calibration
  3. Pipeline B multi-instance inference (Gemini localized crops)
"""

from pathlib import Path
from typing import Union, Tuple, Optional, Dict
import numpy as np
import cv2
import torch
from PIL import Image

DEFAULT_INSTANCE_TARGET_SIZE = (256, 256)
DEFAULT_TARGET_OCCUPANCY_RATIO = 0.88


def canonicalize_instance_crop(
    crop_bgr_or_rgb: np.ndarray,
    target_occupancy_ratio: float = DEFAULT_TARGET_OCCUPANCY_RATIO,
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Transforms a tight product crop into a canonical square frame where the product
    occupies `target_occupancy_ratio` of the frame major dimension, matching the
    framing statistics of the GOOD reference dataset.
    
    Args:
        crop_bgr_or_rgb: HxW product crop array.
        target_occupancy_ratio: Target ratio of product max dimension relative to frame size (default 0.88).
        pad_color: Background padding color.
        
    Returns:
        Square H_can x H_can canonical image array.
    """
    h, w = crop_bgr_or_rgb.shape[:2]
    max_dim = max(h, w)
    
    # Calculate required canonical square canvas dimension
    canvas_dim = max(1, int(round(max_dim / float(target_occupancy_ratio))))
    
    pad_top = (canvas_dim - h) // 2
    pad_bottom = canvas_dim - h - pad_top
    pad_left = (canvas_dim - w) // 2
    pad_right = canvas_dim - w - pad_left
    
    if len(crop_bgr_or_rgb.shape) == 3:
        canonical = cv2.copyMakeBorder(
            crop_bgr_or_rgb,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )
    else:
        canonical = cv2.copyMakeBorder(
            crop_bgr_or_rgb,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color[0]
        )
        
    return canonical


def pad_image_to_square(
    img_bgr_or_rgb: np.ndarray,
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Pads an HxW image symmetrically with a neutral background to make it a 1:1 square.
    Preserves exact aspect ratio and spatial object geometry prior to resizing.
    """
    h, w = img_bgr_or_rgb.shape[:2]
    if h == w:
        return img_bgr_or_rgb.copy()

    max_dim = max(h, w)
    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left

    if len(img_bgr_or_rgb.shape) == 3:
        padded = cv2.copyMakeBorder(
            img_bgr_or_rgb,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )
    else:
        padded = cv2.copyMakeBorder(
            img_bgr_or_rgb,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color[0]
        )

    return padded


def preprocess_instance_crop_to_tensor(
    crop_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = DEFAULT_INSTANCE_TARGET_SIZE,
    pad_to_square: bool = True,
    use_canonical_framing: bool = True,
    target_occupancy_ratio: float = DEFAULT_TARGET_OCCUPANCY_RATIO
) -> torch.Tensor:
    """
    Preprocesses a product instance crop into a PyTorch float32 tensor (3, H, W) scaled [0, 1].
    
    Args:
        crop_input: File path, PIL Image, or BGR/RGB numpy array.
        target_size: Desired output (W, H) tuple (default 256x256).
        pad_to_square: Whether to apply aspect-preserving square padding before resizing.
        use_canonical_framing: Whether to apply canonical framing padding to match GOOD dataset occupancy.
        target_occupancy_ratio: Framing occupancy ratio (default 0.88).
        
    Returns:
        torch.Tensor of shape (3, target_size[1], target_size[0]) float32 in [0.0, 1.0].
    """
    if isinstance(crop_input, (str, Path)):
        img_pil = Image.open(Path(crop_input)).convert("RGB")
        img_np = np.array(img_pil)  # RGB
    elif isinstance(crop_input, Image.Image):
        img_np = np.array(crop_input.convert("RGB"))  # RGB
    elif isinstance(crop_input, np.ndarray):
        if crop_input.ndim == 2:
            img_np = cv2.cvtColor(crop_input, cv2.COLOR_GRAY2RGB)
        elif crop_input.shape[2] == 3:
            # OpenCV loads as BGR by default; convert BGR to RGB
            img_np = cv2.cvtColor(crop_input, cv2.COLOR_BGR2RGB)
        else:
            img_np = crop_input.copy()
    else:
        raise TypeError(f"Unsupported crop_input type '{type(crop_input)}'.")

    # 1. Canonical framing vs standard square padding
    if use_canonical_framing:
        img_np = canonicalize_instance_crop(img_np, target_occupancy_ratio=target_occupancy_ratio, pad_color=(0, 0, 0))
    elif pad_to_square:
        img_np = pad_image_to_square(img_np, pad_color=(0, 0, 0))

    resample_mode = getattr(Image, "Resampling", Image).BILINEAR
    img_pil_resized = Image.fromarray(img_np).resize(target_size, resample_mode)

    # 3. Convert to float32 numpy array in [0, 1]
    arr = np.array(img_pil_resized, dtype=np.float32) / 255.0

    # 4. Convert to PyTorch Tensor with shape (3, H, W)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    return tensor


def generate_physical_screw_mask(
    crop_rgb: np.ndarray,
    bg_color: Tuple[float, float, float] = (202.0, 202.0, 202.0)
) -> np.ndarray:
    """
    Generates a high-precision binary mask identifying the physical screw object pixels inside an instance crop.
    
    Args:
        crop_rgb: HxW x 3 RGB image array.
        bg_color: Target background RGB color (default (202.0, 202.0, 202.0)).
        
    Returns:
        Boolean numpy array (H, W) where True = physical screw object, False = background.
    """
    # 1. Color distance from neutral background
    diff = crop_rgb.astype(np.float32) - np.array(bg_color, dtype=np.float32)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # 2. Otsu thresholding on distance map
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    fg_mask = (binary > 0)
    
    # Fallback if mask is empty or too small
    if np.mean(fg_mask) < 0.05:
        fg_mask = (dist > 20.0)

    return fg_mask


def apply_border_exclusion_to_instance_heatmap_and_bbox(
    crop_path: Path,
    heat_bgr: np.ndarray,
    threshold: float = 24.54
) -> Tuple[np.ndarray, Optional[Dict[str, int]]]:
    """
    Applies physical foreground masking to an instance crop heatmap and recomputes the
    defect localization bounding box using ONLY valid foreground anomaly pixels.
    
    Args:
        crop_path: Path to original instance crop image.
        heat_bgr: HxW x 3 BGR heatmap image array from PatchCore.
        threshold: Production threshold value.
        
    Returns:
        Tuple of (cleaned_heat_bgr, foreground_defect_bbox).
    """
    try:
        crop_pil = Image.open(crop_path).convert("RGB")
        crop_rgb = np.array(crop_pil)
        h, w = crop_rgb.shape[:2]

        if heat_bgr.shape[:2] != (h, w):
            heat_bgr = cv2.resize(heat_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        # 1. Generate physical screw mask
        fg_mask = generate_physical_screw_mask(crop_rgb)

        # 2. Cleaned Heatmap: Mask background pixels outside physical screw
        # Keep background dark / neutral grayscale from crop, overlay colorized heatmap only on foreground screw pixels
        cleaned_heat_bgr = heat_bgr.copy()
        bg_dark_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR) // 3
        cleaned_heat_bgr[~fg_mask] = bg_dark_bgr[~fg_mask]

        # 3. Compute Foreground-Only Defect Localization Bounding Box
        # Convert heat_bgr intensity to normalized anomaly map representation
        heat_gray = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2GRAY)
        
        # Anomaly mask: pixels that are anomalous AND inside physical screw body
        fg_anomaly_mask = (heat_gray > 128) & fg_mask
        fg_anomaly_u8 = (fg_anomaly_mask.astype(np.uint8)) * 255

        contours, _ = cv2.findContours(fg_anomaly_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fg_bbox = None
        if contours:
            from services.localization_service import select_localization_component_intensity_weighted
            fg_bbox = select_localization_component_intensity_weighted(contours, heat_gray)

        return cleaned_heat_bgr, fg_bbox

    except Exception as e:
        print(f"[WARN] Border exclusion processing failed for crop '{crop_path}': {e}")
        return heat_bgr, None

