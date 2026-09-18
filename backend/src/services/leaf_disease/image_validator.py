"""
image_validator.py — Image Quality & Content Validator (Phase 1)
----------------------------------------------------------------
Performs local validation on uploaded leaf images before calling any external API:
  - File format validation (JPEG, PNG, WEBP)
  - File size limits (Max 10MB)
  - Image corruption & header checks
  - Resolution checks (Min 150x150)
  - Blur detection via Laplacian variance
  - Exposure / brightness checks
"""

import io
from typing import List, Tuple
from PIL import Image, ImageStat
import numpy as np

from schemas.leaf_disease import ImageValidationDetail, OverallImageValidationResult


# Threshold constants
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIMENSION = 150
MAX_DIMENSION = 8000
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
MIN_BLUR_VARIANCE = 25.0  # Threshold for blur detection
MIN_BRIGHTNESS = 15.0     # Too dark threshold
MAX_BRIGHTNESS = 245.0    # Too bright/overexposed threshold


def calculate_blur_variance(pil_img: Image.Image) -> float:
    """Calculates Laplacian variance of grayscale image to measure sharpness/blur."""
    try:
        import cv2
        gray = np.array(pil_img.convert("L"))
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return variance
    except ImportError:
        # Fallback using PIL gradient approximation if OpenCV is missing
        gray = pil_img.convert("L")
        stat = ImageStat.Stat(gray)
        # Approximate sharpness via standard deviation of pixel values
        std_dev = stat.stddev[0] if stat.stddev else 0.0
        return float(std_dev * std_dev)


def calculate_brightness(pil_img: Image.Image) -> float:
    """Calculates average brightness of grayscale image (0..255)."""
    gray = pil_img.convert("L")
    stat = ImageStat.Stat(gray)
    return float(stat.mean[0]) if stat.mean else 0.0


def validate_single_image(filename: str, content: bytes) -> ImageValidationDetail:
    errors: List[str] = []
    warnings: List[str] = []
    size_bytes = len(content)

    if size_bytes == 0:
        return ImageValidationDetail(
            filename=filename,
            is_valid=False,
            size_bytes=0,
            errors=["Empty file submitted."]
        )

    if size_bytes > MAX_FILE_SIZE_BYTES:
        errors.append(f"File size exceeds maximum limit of 10MB ({round(size_bytes / (1024*1024), 2)}MB).")

    # Try decoding image
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        # Re-open after verify() because verify destroys PIL image state
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        return ImageValidationDetail(
            filename=filename,
            is_valid=False,
            size_bytes=size_bytes,
            errors=[f"Corrupted image file or invalid format: {str(e)}"]
        )

    fmt = (img.format or "").upper()
    width, height = img.size

    if fmt not in ALLOWED_FORMATS:
        errors.append(f"Unsupported image format '{fmt}'. Allowed formats: JPG, PNG, WEBP.")

    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        errors.append(f"Image resolution {width}x{height} is too low. Minimum required is {MIN_DIMENSION}x{MIN_DIMENSION}.")

    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        warnings.append(f"Image resolution {width}x{height} is very high. It will be resized during processing.")

    # Calculate blur and brightness
    blur_score = calculate_blur_variance(img)
    brightness = calculate_brightness(img)

    if blur_score < MIN_BLUR_VARIANCE:
        warnings.append(f"Image appears blurry (sharpness score: {round(blur_score, 1)}). A clearer close-up is recommended.")

    if brightness < MIN_BRIGHTNESS:
        warnings.append("Image is very dark / underexposed.")
    elif brightness > MAX_BRIGHTNESS:
        warnings.append("Image is overexposed / washed out.")

    is_valid = len(errors) == 0

    return ImageValidationDetail(
        filename=filename,
        is_valid=is_valid,
        format=fmt,
        width=width,
        height=height,
        size_bytes=size_bytes,
        blur_score=round(blur_score, 2),
        brightness=round(brightness, 2),
        errors=errors,
        warnings=warnings
    )


def validate_multiple_images(file_tuples: List[Tuple[str, bytes]]) -> OverallImageValidationResult:
    """Validates 1 to 5 images."""
    details: List[ImageValidationDetail] = []
    valid_count = 0
    invalid_count = 0

    for filename, content in file_tuples:
        detail = validate_single_image(filename, content)
        details.append(detail)
        if detail.is_valid:
            valid_count += 1
        else:
            invalid_count += 1

    return OverallImageValidationResult(
        total_submitted=len(file_tuples),
        valid_count=valid_count,
        invalid_count=invalid_count,
        is_any_valid=valid_count > 0,
        details=details
    )
