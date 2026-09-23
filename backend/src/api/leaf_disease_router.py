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
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Depends, Query
from pymongo.database import Database
from db.connection import get_db

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


def _serialize_leaf_run(doc: dict) -> dict:
    """Helper to convert BSON ObjectId to string for FastAPI responses."""
    if not doc:
        return doc
    res = dict(doc)
    if "_id" in res:
        res["_id"] = str(res["_id"])
        res["id"] = res["_id"]
    return res


@router.post("/analyze", response_model=LeafDiseaseAnalysisResponse)
async def analyze_leaf_disease(
    files: List[UploadFile] = File(...),
    selected_crop: Optional[str] = Form(None),
    db: Database = Depends(get_db)
):
    """
    Phase 1 Leaf Disease Analysis Endpoint.
    Accepts 1 to 5 leaf image files and optional crop selection override.
    Uses Gemini Vision as intelligent fallback when primary providers are uncertain.
    Persists analysis run history to MongoDB.
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
        resp = LeafDiseaseAnalysisResponse(
            analysis_id=analysis_id,
            timestamp=timestamp_str,
            status="FAILED",
            validation=validation_res,
            crop=identify_crop(selected_crop=selected_crop),
            disease=None,
            diagnosis=None,
            retry_guidance="All submitted images failed validation. Please upload clear, uncorrupted images (JPEG, PNG, or WEBP) with adequate lighting.",
        )
        # Save failed run to history
        try:
            doc_data = resp.model_dump()
            doc_data["created_at"] = datetime.now(timezone.utc)
            doc_data["filenames"] = [f[0] for f in file_tuples]
            db.ad_inspections.insert_one(doc_data)
        except Exception as e:
            print(f"[WARN] Failed to save leaf disease run history: {e}")
        return resp

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

    resp = LeafDiseaseAnalysisResponse(
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

    # Save run record to MongoDB leaf_disease_runs collection
    try:
        doc_data = resp.model_dump(mode="json")
        doc_data["created_at"] = datetime.now(timezone.utc)
        doc_data["filenames"] = [f[0] for f in file_tuples]
        db.ad_inspections.insert_one(doc_data)
        print(f"[INFO] Saved leaf disease run '{analysis_id}' to MongoDB history.")
    except Exception as e:
        print(f"[WARN] Failed to persist leaf disease run history: {e}")

    return resp


@router.get("/catalog")
def get_nacl_catalog(db: Database = Depends(get_db)):
    """
    Returns full NACL product catalog organized by category for interactive UI dropdown.
    """
    from db.nacl_product_db import get_all_nacl_products
    products = get_all_nacl_products(db)
    categorized: dict = {}
    for p in products:
        if "_id" in p:
            p["_id"] = str(p["_id"])
        cat = p.get("category", "Other Agrochemicals")
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(p)
    return {
        "total_count": len(products),
        "categories": categorized
    }


@router.post("/feedback")
def submit_leaf_disease_feedback(
    payload: dict = Depends(lambda: None), # allow raw JSON body
    db: Database = Depends(get_db)
):
    """
    Stores user feedback on AI diagnosis to improve Gemini VLM prompt tuning and dataset validation.
    """
    pass


@router.get("/history")
def get_leaf_disease_history(
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None),
    crop_name: Optional[str] = Query(None),
    db: Database = Depends(get_db)
):
    """
    Returns list of historical Leaf Disease Analysis runs ordered newest first.
    Supports filtering by status ('SUCCESS', 'UNCERTAIN', 'FAILED') and crop_name.
    """
    query = {}
    if status:
        query["status"] = status.upper()
    if crop_name:
        query["crop.crop_name"] = crop_name

    cursor = db.ad_inspections.find(query).sort("created_at", -1).limit(limit)
    runs = [_serialize_leaf_run(doc) for doc in cursor]
    return runs


@router.get("/history/{analysis_id}")
def get_leaf_disease_run(analysis_id: str, db: Database = Depends(get_db)):
    """
    Retrieves a single historical Leaf Disease Analysis run by analysis_id.
    """
    doc = db.ad_inspections.find_one({"$or": [{"analysis_id": analysis_id}, {"_id": analysis_id}]})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Leaf disease run with ID '{analysis_id}' not found.")
    return _serialize_leaf_run(doc)


@router.delete("/history/{analysis_id}")
def delete_leaf_disease_run(analysis_id: str, db: Database = Depends(get_db)):
    """
    Deletes a specific Leaf Disease Analysis run record from history.
    """
    res = db.ad_inspections.delete_one({"$or": [{"analysis_id": analysis_id}, {"_id": analysis_id}]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Leaf disease run '{analysis_id}' not found.")
    return {"status": "deleted", "analysis_id": analysis_id}


@router.delete("/history")
def clear_leaf_disease_history(db: Database = Depends(get_db)):
    """
    Clears all historical Leaf Disease Analysis runs.
    """
    res = db.ad_inspections.delete_many({})
    return {"status": "cleared", "deleted_count": res.deleted_count}

