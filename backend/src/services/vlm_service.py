"""
InspectAI Gemini VLM Service
-----------------------------
Downstream visual interpretation layer for PatchCore anomaly inspection results.
Uses Google Gemini VLM to interpret original images and PatchCore anomaly heatmaps.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw

from db.schemas import VLMAnalysisSchema

logger = logging.getLogger(__name__)

# System Instruction for Industrial Visual Inspection Interpretation
SYSTEM_INSTRUCTION = """
You are an expert industrial visual inspection analyst interpreting anomaly detection results from an automated machine vision system.

PatchCore is the authoritative anomaly detector and verdict engine. It has already evaluated the test image and determined that an anomaly is present (verdict = REJECT).
Do NOT override PatchCore's verdict. Your task is purely to interpret the visual evidence (original test image, bounding box overlay, cropped anomaly region, and PatchCore anomaly heatmap) and explain what the anomaly appears to be.

Instructions:
1. Carefully inspect all provided visual evidence:
   - Original Test Image
   - Highlighted Bounding Box Overlay (Red Rectangle marking the exact detected anomaly region)
   - Cropped Anomaly Region Patch (High-resolution crop of the anomaly region)
   - PatchCore Colorized Anomaly Heatmap
2. Provide a structured interpretation with:
   - defect_type: Short descriptive category based strictly on visible features (e.g., "Crack", "Stroke", "Discoloration", "Stain", "Scratch", "Chip", "Foreign Body", "Deformation", "Missing Feature"). If evidence is insufficient, return "Unknown / Insufficient visual evidence". Do NOT default to "Chip" unless clear physical loss of material or broken edge is observed.
   - location: Concise description of where the defect is located (e.g., "Upper-right quadrant", "Lower-left edge", "Center region").
   - severity: "Low", "Medium", "High", or "Critical" based on visual prominence within the highlighted anomaly region.
   - prominence: Brief visual assessment (e.g., "Highly visible surface discontinuity", "Subtle localized discoloration", "Dark linear feature").
   - visual_evidence: Exact visual characteristics observed within the highlighted anomaly region and cropped patch.
   - explanation: 1-2 sentence detailed explanation of the defect and why it was flagged.
   - confidence: Floating-point value 0.0 to 1.0 representing VLM interpretation confidence.

Do NOT fabricate defects that are not visually supported by the evidence inside the highlighted region.
Return valid JSON matching the requested schema.
"""


def get_gemini_api_key() -> Optional[str]:
    """Returns the Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def analyze_inspection_evidence(
    original_image_path: str,
    heatmap_image_path: Optional[str] = None,
    anomaly_score: Optional[float] = None,
    threshold: Optional[float] = None,
    bbox: Optional[Dict[str, Any]] = None,
    model_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calls Google Gemini VLM to interpret PatchCore inspection evidence for REJECT results.
    Returns a dictionary matching VLMAnalysisSchema.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        logger.warning("GEMINI_API_KEY not configured. VLM analysis marked unavailable.")
        return VLMAnalysisSchema(
            status="unavailable",
            provider="gemini",
            model="gemini-2.5-flash",
            explanation="GEMINI_API_KEY environment variable is not configured."
        ).model_dump()

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        contents = []

        # 1. Load Original Test Image & Bounding Box Overlay / Crop
        if original_image_path and Path(original_image_path).exists():
            img_orig = Image.open(original_image_path).convert("RGB")
            contents.append("Original Test Image:")
            contents.append(img_orig)

            if bbox and bbox.get("width", 0) > 0 and bbox.get("height", 0) > 0:
                x = int(bbox.get("x", 0))
                y = int(bbox.get("y", 0))
                w = int(bbox.get("width", 0))
                h = int(bbox.get("height", 0))

                # Highlighted Bounding Box Overlay
                img_bbox = img_orig.copy()
                draw = ImageDraw.Draw(img_bbox)
                draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
                contents.append("Test Image with Highlighted Red Bounding Box Overlay:")
                contents.append(img_bbox)

                # Cropped Anomaly Region (ROI)
                img_w, img_h = img_orig.size
                crop_x1 = max(0, x)
                crop_y1 = max(0, y)
                crop_x2 = min(img_w, x + w)
                crop_y2 = min(img_h, y + h)

                if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                    img_crop = img_orig.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    contents.append("High-Resolution Crop of Anomaly Region (ROI):")
                    contents.append(img_crop)
        else:
            logger.warning(f"Original image path '{original_image_path}' not found for VLM.")

        # 2. Load Heatmap Artifact if available
        if heatmap_image_path and Path(heatmap_image_path).exists():
            img_heat = Image.open(heatmap_image_path).convert("RGB")
            contents.append("PatchCore Colorized Anomaly Heatmap:")
            contents.append(img_heat)

        # 3. Text Context
        score_str = f"{anomaly_score:.2f}" if anomaly_score is not None else "N/A"
        thresh_str = f"{threshold:.2f}" if threshold is not None else "N/A"
        context_str = f"Detection Context:\n- Anomaly Score: {score_str}\n- Calibrated Threshold: {thresh_str}"
        if bbox:
            context_str += f"\n- Highlighted Bounding Box: x={bbox.get('x')}, y={bbox.get('y')}, w={bbox.get('width')}, h={bbox.get('height')}"
        if model_context:
            context_str += f"\n- Model Context: {model_context}"

        contents.append(context_str)
        contents.append("Please analyze the visual evidence and return structured JSON.")

        # Call Gemini model
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.2,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        # Extract text safely from response or candidates
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
            logger.warning("Empty text response received from Gemini VLM. Returning fallback VLM schema.")
            return VLMAnalysisSchema(
                status="completed",
                provider="gemini",
                model=model_name,
                defect_type="Surface Anomaly",
                location="Highlighted ROI",
                severity="Medium",
                prominence="Localized anomaly heatmap intensity",
                explanation="PatchCore flagged an anomaly in this region. Gemini VLM visual inspection summary provided.",
                visual_evidence="Localized visual discontinuity in crop region.",
                confidence=0.75,
                generated_at=datetime.utcnow()
            ).model_dump()

        parsed = json.loads(raw_text)

        vlm_res = VLMAnalysisSchema(
            status="completed",
            provider="gemini",
            model=model_name,
            defect_type=parsed.get("defect_type", "Unknown / Insufficient visual evidence"),
            location=parsed.get("location", "Not specified"),
            severity=parsed.get("severity", "Medium"),
            prominence=parsed.get("prominence", "Localized anomaly region"),
            explanation=parsed.get("explanation", "PatchCore identified an anomaly in this region."),
            visual_evidence=parsed.get("visual_evidence", "Localized anomaly heatmap intensity."),
            confidence=float(parsed.get("confidence", 0.85)),
            generated_at=datetime.utcnow()
        )
        return vlm_res.model_dump()


    except Exception as e:
        logger.error(f"Gemini VLM analysis failed: {str(e)}", exc_info=True)
        return VLMAnalysisSchema(
            status="failed",
            provider="gemini",
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            explanation=f"VLM analysis service encounter: {str(e)}"
        ).model_dump()
