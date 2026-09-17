"""
InspectAI Gemini Instance Localization Service
-----------------------------------------------
Dedicated service that uses Google Gemini VLM solely to identify distinct
physical product instances in composite test images for downstream PatchCore inspection.

Gemini is ONLY an instance localizer:
- Returns 0-1000 normalized bounding boxes for distinct physical objects.
- Does NOT perform anomaly detection, verdict determination, or score calculation.
- PatchCore remains the sole anomaly detector.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

# Configurable Gemini Model via environment variable (default: gemini-2.5-flash)
def get_gemini_instance_model() -> str:
    return os.getenv("GEMINI_INSTANCE_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_gemini_api_key() -> Optional[str]:
    """Returns Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


SYSTEM_INSTRUCTION = """
You are a physical object localization system for industrial visual inspection.
Your ONLY job is to identify distinct physical product instances visible in an image for downstream PatchCore anomaly inspection.
Do NOT attempt to perform anomaly detection, verdict determination, score calculation, or defect diagnosis.

Rules:
1. Detect every distinct physical product visible in the image.
2. If multiple identical products are present, return one bounding box per physical product.
3. Do NOT split one product into multiple regions.
4. Do NOT treat cracks, stains, texture, holes, scratches, shadows, reflections, color variations, or surface patterns as separate products.
5. Do NOT treat grid cells or visual texture regions as products.
6. Do NOT detect background or decorative regions.
7. Return a practical bounding box around the COMPLETE physical product.
8. Do not invent products when boundaries are uncertain.
9. If no distinct product is visible, return an empty instances list.
10. IDs must be unique and sequential starting at 1.
11. Return ONLY the required structured JSON matching the requested schema:
{
  "instances": [
    {
      "id": 1,
      "label": "product",
      "box_2d": [ymin, xmin, ymax, xmax]
    }
  ]
}
Coordinates in box_2d must be integers normalized to 0-1000 [ymin, xmin, ymax, xmax].
"""


def compute_iou_normalized(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes IoU for two normalized boxes [ymin, xmin, ymax, xmax]."""
    yA1, xA1, yA2, xA2 = boxA
    yB1, xB1, yB2, xB2 = boxB

    inter_y1 = max(yA1, yB1)
    inter_x1 = max(xA1, xB1)
    inter_y2 = min(yA2, yB2)
    inter_x2 = min(xA2, xB2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = (yA2 - yA1) * (xA2 - xA1)
    areaB = (yB2 - yB1) * (xB2 - xB1)
    union_area = areaA + areaB - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def validate_and_deduplicate_boxes(
    raw_instances: List[Dict[str, Any]],
    orig_width: int,
    orig_height: int,
    iou_threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Validates, filters, deduplicates, and converts normalized 0-1000 box_2d
    into pixel coordinates on the original image.
    """
    valid_candidates = []

    for item in raw_instances:
        box_2d = item.get("box_2d")
        if not box_2d or not isinstance(box_2d, list) or len(box_2d) != 4:
            continue

        try:
            ymin, xmin, ymax, xmax = [int(v) for v in box_2d]
        except (ValueError, TypeError):
            continue

        # 1. Bounds check
        ymin = max(0, min(1000, ymin))
        xmin = max(0, min(1000, xmin))
        ymax = max(0, min(1000, ymax))
        xmax = max(0, min(1000, xmax))

        if ymin >= ymax or xmin >= xmax:
            continue

        w_norm = xmax - xmin
        h_norm = ymax - ymin
        area_norm = w_norm * h_norm

        # 2. Minimum dimension/area filters (must be >= 1.5% dim, >= 0.03% total area)
        if w_norm < 15 or h_norm < 15 or area_norm < 300:
            continue

        # 3. Convert to pixel coordinates on original image
        px = max(0, int(round((xmin / 1000.0) * orig_width)))
        py = max(0, int(round((ymin / 1000.0) * orig_height)))
        pw = min(orig_width - px, int(round((w_norm / 1000.0) * orig_width)))
        ph = min(orig_height - py, int(round((h_norm / 1000.0) * orig_height)))

        if pw <= 0 or ph <= 0:
            continue

        valid_candidates.append({
            "box_2d": [ymin, xmin, ymax, xmax],
            "pixel_bbox": {"x": px, "y": py, "width": pw, "height": ph},
            "area_norm": area_norm,
            "label": item.get("label", "product")
        })

    # Sort candidates by normalized area descending
    valid_candidates.sort(key=lambda b: b["area_norm"], reverse=True)

    # 4. IoU Deduplication
    keep_boxes = []
    for cand in valid_candidates:
        boxA = cand["box_2d"]
        duplicate = False
        for kept in keep_boxes:
            boxB = kept["box_2d"]
            if compute_iou_normalized(boxA, boxB) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            keep_boxes.append(cand)

    # Re-assign sequential IDs starting at 1
    final_instances = []
    for idx, cand in enumerate(keep_boxes, start=1):
        final_instances.append({
            "id": idx,
            "label": cand["label"],
            "box_2d": cand["box_2d"],
            "pixel_bbox": cand["pixel_bbox"]
        })

    return final_instances


def localize_product_instances(
    image_path: str,
    model_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calls Google Gemini VLM to localize distinct physical product instances in an image.
    Returns structured localization results with original image pixel coordinates.
    """
    start_time = time.time()
    api_key = get_gemini_api_key()
    model_name = get_gemini_instance_model()

    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Gemini instance localization unavailable.")
        return {
            "status": "failed",
            "reason": "GEMINI_INSTANCE_LOCALIZATION_FAILED",
            "error": "GEMINI_API_KEY environment variable is not configured.",
            "instances": [],
            "raw_count": 0,
            "validated_count": 0,
            "latency_ms": 0.0,
            "model": model_name
        }

    if not Path(image_path).exists():
        logger.error(f"Image path '{image_path}' does not exist for Gemini instance localization.")
        return {
            "status": "failed",
            "reason": "IMAGE_FILE_NOT_FOUND",
            "error": f"Image file not found: {image_path}",
            "instances": [],
            "raw_count": 0,
            "validated_count": 0,
            "latency_ms": 0.0,
            "model": model_name
        }

    try:
        pil_img = Image.open(image_path).convert("RGB")
        orig_width, orig_height = pil_img.size

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        user_prompt = "Find all distinct physical objects in this image that should each be inspected independently by PatchCore anomaly detection. Return structured JSON."
        if model_context:
            user_prompt += f" Product context: {model_context}."

        contents = [pil_img, user_prompt]

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.1,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

        latency_ms = (time.time() - start_time) * 1000.0

        raw_text = None
        if response and hasattr(response, "text") and response.text:
            raw_text = response.text
        elif response and hasattr(response, "candidates") and response.candidates:
            for cand in response.candidates:
                if cand.content and cand.content.parts:
                    part_texts = [p.text for p in cand.content.parts if hasattr(p, "text") and p.text]
                    if part_texts:
                        raw_text = "".join(part_texts)
                        break

        if not raw_text:
            logger.warning("Empty text response received from Gemini instance localization.")
            return {
                "status": "failed",
                "reason": "GEMINI_INSTANCE_LOCALIZATION_FAILED",
                "error": "Gemini API returned an empty response.",
                "instances": [],
                "raw_count": 0,
                "validated_count": 0,
                "latency_ms": round(latency_ms, 2),
                "model": model_name
            }

        parsed = json.loads(raw_text)
        raw_instances = parsed.get("instances", [])
        if not isinstance(raw_instances, list):
            raw_instances = []

        validated_instances = validate_and_deduplicate_boxes(raw_instances, orig_width, orig_height)

        raw_box_2ds = [inst.get("box_2d") for inst in raw_instances if isinstance(inst, dict)]
        val_bboxes = [inst.get("pixel_bbox") for inst in validated_instances]

        logger.info(
            f"=== GEMINI DIAGNOSTIC LOG ===\n"
            f"Model: {model_name}\n"
            f"Image Dimensions: {orig_width}x{orig_height}\n"
            f"Raw Count: {len(raw_instances)}\n"
            f"Raw box_2d: {raw_box_2ds}\n"
            f"Validated Count: {len(validated_instances)}\n"
            f"Validated pixel_bbox: {val_bboxes}\n"
            f"=============================="
        )
        print(
            f"\n=== GEMINI DIAGNOSTIC LOG ===\n"
            f"Model: {model_name}\n"
            f"Image Dimensions: {orig_width}x{orig_height}\n"
            f"Raw Count: {len(raw_instances)}\n"
            f"Raw box_2d: {raw_box_2ds}\n"
            f"Validated Count: {len(validated_instances)}\n"
            f"Validated pixel_bbox: {val_bboxes}\n"
            f"==============================\n",
            flush=True
        )

        logger.info(
            f"Gemini instance localization completed for '{image_path}': "
            f"model={model_name}, raw_count={len(raw_instances)}, "
            f"validated_count={len(validated_instances)}, latency_ms={latency_ms:.2f}ms"
        )

        if not validated_instances:
            return {
                "status": "no_instances",
                "reason": "NO_PRODUCT_INSTANCES_DETECTED",
                "instances": [],
                "raw_count": len(raw_instances),
                "validated_count": 0,
                "latency_ms": round(latency_ms, 2),
                "model": model_name
            }

        return {
            "status": "success",
            "reason": None,
            "instances": validated_instances,
            "raw_count": len(raw_instances),
            "validated_count": len(validated_instances),
            "latency_ms": round(latency_ms, 2),
            "model": model_name
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000.0
        logger.error(f"Gemini instance localization failed for '{image_path}': {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "reason": "GEMINI_INSTANCE_LOCALIZATION_FAILED",
            "error": str(e),
            "instances": [],
            "raw_count": 0,
            "validated_count": 0,
            "latency_ms": round(latency_ms, 2),
            "model": model_name
        }
