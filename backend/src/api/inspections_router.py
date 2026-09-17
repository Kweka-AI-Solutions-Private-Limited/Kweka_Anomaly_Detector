"""
InspectAI Inspections API Router
--------------------------------
Endpoints for running single-sample inspections, listing inspection history,
fetching inspection details, and submitting feedback.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel

from db.connection import get_db
from services.inspection_service import (
    run_inspection, get_inspections, get_inspection,
    submit_feedback, get_feedback, get_all_feedback, retry_vlm_analysis,
    retry_instance_vlm_analysis
)

router = APIRouter(prefix="/inspections", tags=["Inspections"])



class FeedbackCreateRequest(BaseModel):
    detection_feedback: str = "correct"  # correct, false_positive, false_negative
    vlm_feedback_categories: Optional[List[str]] = None  # correct, wrong_defect_type, wrong_location, wrong_severity
    corrected_defect_type: Optional[str] = None
    corrected_location: Optional[str] = None
    corrected_severity: Optional[str] = None
    comment: Optional[str] = None
    feedback_type: Optional[str] = None  # backward compatibility


@router.post("", status_code=201)
def create_inspection_endpoint(
    model_id: str = Form(...),
    threshold_override: Optional[float] = Form(None),
    inspection_mode: str = Form("single_image"),
    min_instance_area: int = Form(500),
    max_instances: int = Form(20),
    image: UploadFile = File(...),
    db=Depends(get_db)
):
    """
    Runs PatchCore anomaly detection inference on an uploaded test image.
    Supports single product mode (default) and multi_instance mode.
    """
    return run_inspection(
        db,
        model_id=model_id,
        upload_file=image,
        threshold_override=threshold_override,
        inspection_mode=inspection_mode,
        min_instance_area=min_instance_area,
        max_instances=max_instances
    )



@router.get("")
def list_inspections_endpoint(
    model_id: Optional[str] = Query(None),
    model_version_id: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    defect_type: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1),
    db=Depends(get_db)
):
    """
    Lists inspection history with comprehensive filtering
    (model_id, model_version_id, run_id, status, defect_type, location, severity, date range, search, pagination).
    """
    return get_inspections(
        db,
        model_id=model_id,
        model_version_id=model_version_id,
        run_id=run_id,
        status=status,
        defect_type=defect_type,
        location=location,
        severity=severity,
        start_date=start_date,
        end_date=end_date,
        search=search,
        skip=skip,
        limit=limit
    )


@router.get("/feedback/workspace")
def list_workspace_feedback_endpoint(
    model_id: Optional[str] = Query(None),
    model_version_id: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    feedback_type: Optional[str] = Query(None),
    db=Depends(get_db)
):
    """Lists aggregated feedback records for the Feedback Workspace."""
    return get_all_feedback(
        db,
        model_id=model_id,
        model_version_id=model_version_id,
        run_id=run_id,
        feedback_type=feedback_type
    )


@router.get("/{inspection_id}")
def get_inspection_endpoint(inspection_id: str, db=Depends(get_db)):
    """Returns details for a single inspection record + result."""
    return get_inspection(db, inspection_id)


@router.post("/{inspection_id}/analyze-vlm")
def analyze_vlm_endpoint(inspection_id: str, force: bool = Query(False), db=Depends(get_db)):
    """Triggers or retries Gemini VLM analysis for an existing REJECT inspection."""
    return retry_vlm_analysis(db, inspection_id, force=force)


@router.post("/{inspection_id}/instances/{instance_id}/analyze-vlm")
def analyze_instance_vlm_endpoint(
    inspection_id: str,
    instance_id: int,
    force: bool = Query(False),
    db=Depends(get_db)
):
    """Triggers or retries Gemini VLM analysis for a specific REJECT instance in a multi-instance inspection."""
    return retry_instance_vlm_analysis(db, inspection_id=inspection_id, instance_id=instance_id, force=force)



@router.post("/{inspection_id}/feedback", status_code=201)
def submit_feedback_endpoint(inspection_id: str, req: FeedbackCreateRequest, db=Depends(get_db)):
    """Submits user feedback for an inspection."""
    return submit_feedback(
        db,
        inspection_id=inspection_id,
        detection_feedback=req.detection_feedback,
        vlm_feedback_categories=req.vlm_feedback_categories,
        corrected_defect_type=req.corrected_defect_type,
        corrected_location=req.corrected_location,
        corrected_severity=req.corrected_severity,
        comment=req.comment,
        feedback_type=req.feedback_type
    )


@router.get("/{inspection_id}/feedback")
def get_feedback_endpoint(inspection_id: str, db=Depends(get_db)):
    """Lists feedback records for an inspection."""
    return get_feedback(db, inspection_id)

