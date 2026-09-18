"""
leaf_disease_router.py — Phase 1 FastAPI Endpoint for Leaf Disease Analysis
-----------------------------------------------------------------------------
Orchestrates:
  1. Image Validation
  2. Crop / Plant Identification  (+ Gemini Vision fallback when UNCERTAIN)
  3. Primary Disease Detection    (+ Gemini Vision fallback when LOW confidence)
  4. Diagnosis Validation
  5. Response Formatting
"""

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status

from schemas.leaf_disease import LeafDiseaseAnalysisResponse
from services.leaf_disease.image_validator import validate_multiple_images
from services.leaf_disease.plant_identifier import identify_crop
from services.leaf_disease.disease_provider import get_disease_provider
from services.leaf_disease.diagnosis_validator import validate_diagnosis_evidence
from services.leaf_disease.gemini_leaf_fallback import gemini_identify_crop, gemini_analyze_disease
from services.leaf_disease.nacl_recommendation_engine import recommendation_engine

router = APIRouter(prefix="/leaf-disease", tags=["Leaf Disease Analysis"])

# ------------------------------------------------------------------
# Gemini fallback thresholds
# UNCERTAIN crop:  status is UNCERTAIN and crop_name is "Unknown"
# LOW disease:     provider confidence below this threshold
# ------------------------------------------------------------------
GEMINI_DISEASE_CONFIDENCE_THRESHOLD = 0.15


@router.post("/analyze", response_model=LeafDiseaseAnalysisResponse)
async def analyze_leaf_disease(
    files: List[UploadFile] = File(...),
    selected_crop: Optional[str] = Form(None)
):
    """
    Phase 1 Leaf Disease Analysis Endpoint.
    Accepts 1 to 5 leaf image files and optional crop selection override.
    Uses Gemini Vision as intelligent fallback when primary providers are uncertain.
    """
    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one leaf image file must be uploaded."
        )

    if len(files) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 5 images allowed per analysis request."
        )

    # Read image contents into memory
    file_tuples = []
    for upload_file in files:
        content = await upload_file.read()
        filename = upload_file.filename or "uploaded_leaf.jpg"
        file_tuples.append((filename, content))

    # 1. Image Validation
    validation_res = validate_multiple_images(file_tuples)

    analysis_id = f"LDA-{uuid.uuid4().hex[:8].upper()}"
    timestamp_str = datetime.now().isoformat()

    # Handle scenario where ALL images are invalid
    if not validation_res.is_any_valid:
        return LeafDiseaseAnalysisResponse(
            analysis_id=analysis_id,
            timestamp=timestamp_str,
            status="FAILED",
            validation=validation_res,
            crop=identify_crop(selected_crop=selected_crop),
            disease=None,
            diagnosis=None,
            retry_guidance="All submitted images failed validation. Please upload clear, uncorrupted images (JPEG, PNG, or WEBP) with adequate lighting.",
        )

    # Filter details for usable valid images
    valid_details = [d for d in validation_res.details if d.is_valid]
    valid_filenames = [d.filename for d in valid_details]
    valid_raw_files = [(fn, fb) for fn, fb in file_tuples if any(fn == d.filename for d in valid_details)]

    # 2. Crop Identification
    crop_info = identify_crop(selected_crop=selected_crop, valid_filenames=valid_filenames)

    # 2b. Gemini Fallback: crop not identified by primary method
    if crop_info.status == "UNCERTAIN" and crop_info.crop_name == "Unknown":
        gemini_crop = gemini_identify_crop(raw_files=valid_raw_files)
        if gemini_crop is not None:
            crop_info = gemini_crop

    # 3. Disease Detection Provider Call
    provider = get_disease_provider()
    disease_result = provider.detect_disease(
        crop_name=crop_info.crop_name,
        image_details=valid_details,
        raw_files=file_tuples
    )

    # 3b. Gemini Fallback: disease provider returned low confidence or error
    needs_gemini_disease = (
        disease_result.provider_status == "PROVIDER_ERROR"
        or disease_result.confidence is None
        or (disease_result.confidence is not None and disease_result.confidence < GEMINI_DISEASE_CONFIDENCE_THRESHOLD)
    )

    if needs_gemini_disease:
        gemini_disease = gemini_analyze_disease(
            crop_name=crop_info.crop_name,
            raw_files=valid_raw_files
        )
        if gemini_disease is not None:
            disease_result = gemini_disease

    # 4. Diagnosis Validation
    diagnosis_res = validate_diagnosis_evidence(
        validation_result=validation_res,
        crop_info=crop_info,
        disease_result=disease_result
    )

    # Determine overall status & retry guidance
    if disease_result.provider_status == "PROVIDER_ERROR":
        analysis_status = "FAILED"
        retry_guidance = "Disease detection provider encountered an error. Please verify server configuration."
    elif diagnosis_res.is_uncertain:
        analysis_status = "UNCERTAIN"
        retry_guidance = "Diagnosis evidence is uncertain. Please upload a clear close-up of the leaf symptom or select the exact crop species."
    else:
        analysis_status = "SUCCESS"
        retry_guidance = None

    # 5. Phase 3: RAG & Gemini Product Recommendation Engine Call
    nacl_recs = None
    if disease_result and disease_result.disease_name:
        is_healthy_leaf = "healthy" in disease_result.disease_name.lower()
        nacl_recs = recommendation_engine.recommend_products(
            crop_name=crop_info.crop_name,
            disease_name=disease_result.disease_name,
            is_healthy=is_healthy_leaf,
            crop_compatibility_status=diagnosis_res.crop_compatibility_status,
            is_uncertain=diagnosis_res.is_uncertain
        )

    return LeafDiseaseAnalysisResponse(
        analysis_id=analysis_id,
        timestamp=timestamp_str,
        status=analysis_status,
        validation=validation_res,
        crop=crop_info,
        disease=disease_result,
        diagnosis=diagnosis_res,
        nacl_recommendations=nacl_recs,
        retry_guidance=retry_guidance,
    )
