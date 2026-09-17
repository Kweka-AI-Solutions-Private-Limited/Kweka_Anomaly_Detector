"""
InspectAI Inspection Runs API Router
------------------------------------
Endpoints for batch Inspection Runs (creation, listing, run details).
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query, BackgroundTasks

from db.connection import get_db
from services.inspection_run_service import (
    create_inspection_run, get_inspection_runs, get_inspection_run
)

router = APIRouter(prefix="/inspection-runs", tags=["Inspection Runs"])


@router.post("", status_code=201)
def create_inspection_run_endpoint(
    background_tasks: BackgroundTasks,
    model_id: str = Form(...),
    threshold_override: Optional[float] = Form(None),
    inspection_mode: str = Form("single"),
    min_instance_area: int = Form(500),
    max_instances: int = Form(20),
    files: List[UploadFile] = File(...),
    db=Depends(get_db)
):
    """
    Creates and executes an asynchronous batch Inspection Run for multiple test images
    under an ACTIVE model version. Accepts optional threshold_override and inspection_mode.
    Returns run_id immediately and processes images in background.
    """
    return create_inspection_run(
        db,
        model_id=model_id,
        upload_files=files,
        background_tasks=background_tasks,
        threshold_override=threshold_override,
        inspection_mode=inspection_mode,
        min_instance_area=min_instance_area,
        max_instances=max_instances
    )




@router.get("")
def list_inspection_runs_endpoint(
    model_id: Optional[str] = Query(None),
    model_version_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db=Depends(get_db)
):
    """Lists inspection runs with optional filtering (model_id, model_version_id, status)."""
    return get_inspection_runs(db, model_id=model_id, model_version_id=model_version_id, status=status)


@router.get("/{run_id}")
def get_inspection_run_endpoint(run_id: str, db=Depends(get_db)):
    """Returns detailed InspectionRun summary and associated inspection results."""
    return get_inspection_run(db, run_id=run_id)
