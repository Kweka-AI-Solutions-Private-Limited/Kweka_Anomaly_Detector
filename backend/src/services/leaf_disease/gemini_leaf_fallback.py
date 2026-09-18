"""
gemini_leaf_fallback.py — Gemini Vision Fallback for Leaf Disease Pipeline (Phase 1+)
---------------------------------------------------------------------------------------
Provides targeted Gemini VLM calls when primary providers fail or return low-confidence:
  - Crop identification fallback: When auto-detect returns UNCERTAIN/Unknown
  - Disease enrichment fallback: When primary disease provider returns LOW evidence

STRICT RULES:
  - Never used as primary provider. Always secondary/fallback only.
  - Returns structured, normalized results matching existing schemas.
  - Zero NACL product matching, zero treatment recommendations.
  - Never logs or exposes the API key.
"""

import os
import json
import logging
from typing import List, Optional, Tuple, Any, Dict

from schemas.leaf_disease import CropIdentificationResult, NormalizedDiseaseResult, DiseaseCandidate, ImageValidationDetail

logger = logging.getLogger(__name__)

CROP_IDENTIFY_PROMPT = """You are an expert agricultural botanist specializing in crop identification for plant disease diagnosis.

You are given one or more leaf images. Your task is to identify the crop or plant species shown based on strict morphological features.

Botanical Identification Guidelines:
- Cucurbitaceae (Cucumber, Melon, Squash, Pumpkin, Gourd): Broad heart-shaped (cordate) or palmate leaves with shallow rounded lobes, coarse or hairy leaf texture, trailing herbaceous green vines.
- Grape (Vitis vinifera): Deeply palmately lobed leaves with sharp, jagged serrated margins, characteristic palmate venation, woody climbing vines/canes.
- Rosaceae (Apple, Pear): Simple oval/elliptical leaves with fine serration on woody branches.
- Solanaceae (Tomato, Potato, Pepper): Compound or deeply lobed herbaceous leaves.

Instructions:
1. Examine leaf morphology, lobe depth, margin serrations, surface texture, and vine structure carefully.
2. Return ONLY the most likely crop name from this list if it matches: Cucurbits, Tomato, Rice / Paddy, Cotton, Chilli / Pepper, Maize / Corn, Potato, Grape, Apple, Pear, Soybean, Wheat, Sugarcane, Groundnut / Peanut.
3. If the crop is clearly identifiable as cucumber, melon, or squash, return "Cucurbits".
4. If you cannot confidently identify the crop from visual evidence, return "Unknown".
5. Do NOT guess a crop based on disease symptoms. Crop identification must be based on leaf morphology alone.
6. Return ONLY valid JSON — no markdown, no explanation wrapper.

Return this exact JSON format:
{
  "crop_name": "<common crop name>",
  "confidence": <float 0.0-1.0>,
  "evidence_note": "<1 sentence explaining the morphological visual evidence>"
}"""

DISEASE_ANALYSIS_PROMPT = """You are an expert plant pathologist specializing in agricultural crop diseases and plant diagnostics.

You are given leaf image(s) and crop species context. Your task is to identify visible disease symptoms, pests, or nutritional deficiencies accurately based on strict botanical pathology criteria.

CRITICAL DIAGNOSTIC RULES:
1. Examine lesion color, shape, and micro-structures carefully:
   - BRIGHT ORANGE, RUST-RED, or YELLOW circular patches with tiny central black dots (pycnia/spermogonia) on leaves (especially Pear, Apple, or Rosaceae) = RUST FUNGUS (e.g. "European Pear Rust (Gymnosporangium sabinae)" on Pear, or "Cedar-Apple Rust (Gymnosporangium juniperi-virginianae)" on Apple). DO NOT confuse bright orange rust spots with Black Rot!
   - CIRCULAR BROWN TO BLACK necrotic spots ("frog-eye" pattern) or mummies = BLACK ROT (Diplodia seriata / Botryosphaeria obtusa).
   - WHITE/GREY POWDERY COATING on leaf surface = POWDERY MILDEW.
   - GREY/PURPLE DOWNY GROWTH on undersurface with yellowing upper leaf = DOWNY MILDEW.
2. Cross-reference symptoms against the specific host crop species provided.
3. Express all disease names strictly as: "Common name (Scientific name)" — e.g. "European Pear Rust (Gymnosporangium sabinae)".
4. If the leaf appears healthy with no symptoms, return "Healthy — No Visible Disease".
5. Return ONLY valid JSON — no markdown, no explanation wrapper.

Return this exact JSON format:
{
  "disease_name": "<Common name (Scientific name)>",
  "confidence": <float 0.0-1.0>,
  "is_healthy": <true/false>,
  "candidates": [
    {"name": "<2nd condition>", "confidence": <float>},
    {"name": "<3rd condition>", "confidence": <float>}
  ],
  "evidence_note": "<1-2 sentences describing the specific visual evidence (e.g. bright orange rust spots with pycnia)>"
}"""


def _get_gemini_client():
    """Returns configured Gemini client using existing project API key."""
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            return None, None
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        client = genai.Client(api_key=api_key)
        return client, model_name
    except ImportError:
        logger.error("google-genai package not installed.")
        return None, None


def gemini_identify_crop(
    raw_files: List[Tuple[str, bytes]]
) -> Optional[CropIdentificationResult]:
    """
    Calls Gemini Vision to identify crop from leaf images.
    Called ONLY when primary auto-detection returns UNCERTAIN.
    Returns CropIdentificationResult or None if Gemini is unavailable/fails.
    """
    client, model_name = _get_gemini_client()
    if not client:
        logger.warning("Gemini crop identification fallback skipped: GEMINI_API_KEY not configured.")
        return None

    try:
        from google.genai import types
        from PIL import Image
        import io

        contents: List[Any] = [CROP_IDENTIFY_PROMPT]
        for filename, img_bytes in raw_files[:3]:  # Max 3 images to Gemini
            mime = "image/jpeg"
            if filename.lower().endswith(".png"):
                mime = "image/png"
            elif filename.lower().endswith(".webp"):
                mime = "image/webp"
            part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
            contents.append(part)

        if len(contents) < 2:
            return None

        logger.info("Calling Gemini crop identification fallback with %d images.", len(raw_files))
        response = client.models.generate_content(
            model=model_name or "gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        raw_text = response.text or ""
        parsed = json.loads(raw_text.strip())

        crop_name = parsed.get("crop_name", "Unknown").strip()
        confidence = float(parsed.get("confidence", 0.0))
        evidence_note = parsed.get("evidence_note", "Identified by Gemini Vision VLM.")

        if not crop_name or crop_name.lower() == "unknown":
            return CropIdentificationResult(
                crop_name="Unknown",
                confidence=0.0,
                source="gemini_fallback",
                status="UNCERTAIN",
                evidence_note="Gemini Vision could not identify the crop with confidence."
            )

        return CropIdentificationResult(
            crop_name=crop_name,
            confidence=confidence,
            source="gemini_fallback",
            status="CONFIRMED" if confidence >= 0.60 else "UNCERTAIN",
            evidence_note=f"[Gemini Vision] {evidence_note}"
        )

    except Exception as e:
        logger.error("Gemini crop identification fallback failed: %s", str(e))
        return None


def gemini_analyze_disease(
    crop_name: str,
    raw_files: List[Tuple[str, bytes]]
) -> Optional[NormalizedDiseaseResult]:
    """
    Calls Gemini Vision to analyze disease symptoms from leaf images.
    Called ONLY when primary disease provider returns LOW confidence or error.
    Returns NormalizedDiseaseResult or None if Gemini is unavailable/fails.
    """
    client, model_name = _get_gemini_client()
    if not client:
        logger.warning("Gemini disease fallback skipped: GEMINI_API_KEY not configured.")
        return None

    try:
        from google.genai import types
        from PIL import Image
        import io

        prompt = DISEASE_ANALYSIS_PROMPT
        if crop_name and crop_name.lower() != "unknown":
            prompt = f"Crop context: This is a {crop_name} leaf.\n\n" + DISEASE_ANALYSIS_PROMPT

        contents: List[Any] = [prompt]
        for filename, img_bytes in raw_files[:3]:
            mime = "image/jpeg"
            if filename.lower().endswith(".png"):
                mime = "image/png"
            elif filename.lower().endswith(".webp"):
                mime = "image/webp"
            part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
            contents.append(part)

        if len(contents) < 2:
            return None

        logger.info("Calling Gemini disease analysis fallback for crop '%s' with %d images.", crop_name, len(raw_files))
        response = client.models.generate_content(
            model=model_name or "gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.15
            )
        )

        raw_text = response.text or ""
        parsed = json.loads(raw_text.strip())

        disease_name = parsed.get("disease_name", "Unknown Condition").strip()
        confidence = float(parsed.get("confidence", 0.0))
        is_healthy = bool(parsed.get("is_healthy", False))
        evidence_note = parsed.get("evidence_note", "Gemini Vision analysis.")

        raw_candidates = parsed.get("candidates", [])
        candidates = []
        for c in raw_candidates[:4]:
            c_name = c.get("name", "")
            c_conf = float(c.get("confidence", 0.0))
            if c_name:
                candidates.append(DiseaseCandidate(disease_name=c_name, confidence=c_conf))

        logger.info(
            "Gemini disease fallback result: '%s' (conf=%.2f, healthy=%s)",
            disease_name, confidence, is_healthy
        )

        return NormalizedDiseaseResult(
            disease_name=disease_name,
            confidence=confidence,
            provider_name="Gemini Vision (Fallback)",
            is_mock=False,
            supported=True,
            candidates=candidates,
            provider_status="OK",
            error_message=None
        )

    except Exception as e:
        logger.error("Gemini disease analysis fallback failed: %s", str(e))
        return None
