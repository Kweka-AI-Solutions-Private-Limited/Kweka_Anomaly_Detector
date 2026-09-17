"""
InspectAI Inspection Run Service
---------------------------------
Business logic for managing chronological, per-model Inspection Runs.
Hierarchy: Model -> Inspection Run -> Inspection -> Result
"""

import io
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from bson import ObjectId
from fastapi import HTTPException, UploadFile
from pymongo.database import Database

from db.schemas import (
    InspectionRunSchema, RunSummary, InspectionSchema, InspectionInput,
    InspectionResultSchema, PredictionOutput, LocalizationOutput, BoundingBox,
    VLMAnalysisSchema
)
from services.storage_service import save_inspection_image, get_storage_base_dir
from services.patchcore_service import run_patchcore_inference
from services.model_service import get_model, get_model_version
from services.inspection_service import (
    serialize_object_ids, normalize_inspection_payload, run_inspection,
    run_single_image_inspection, run_multi_instance_inspection
)
from services.vlm_service import analyze_inspection_evidence
from services.notification_service import create_run_notification



def get_next_run_number(db: Database, model_id: str) -> int:
    """
    Returns the next sequential, chronological run_number for a specific model.
    Sequence starts at 1 per model.
    """
    latest_run = db.inspection_runs.find_one(
        {"model_id": ObjectId(model_id)},
        sort=[("run_number", -1)]
    )
    if latest_run and "run_number" in latest_run:
        return latest_run["run_number"] + 1
    return 1


from fastapi import BackgroundTasks

class BytesUploadFile:
    """Helper wrapper holding filename and raw bytes for background processing."""
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.file = io.BytesIO(content)


import math

def _process_inspection_run_background(
    db: Database,
    run_id: str,
    model_id: str,
    active_version_id: str,
    version: Dict[str, Any],
    model: Dict[str, Any],
    file_payloads: List[Tuple[str, bytes]],
    threshold_override: Optional[float] = None,
    inspection_mode: str = "single",
    min_instance_area: int = 500,
    max_instances: int = 20
):

    """
    Background worker processing test images sequentially, saving inspection results,
    and atomically incrementing completed_images after EACH image finishes.
    """
    completed_images = 0
    pass_count = 0
    reject_count = 0
    error_count = 0
    inspection_records = []

    if threshold_override is not None:
        try:
            val = float(threshold_override)
            if math.isnan(val) or math.isinf(val) or val <= 0:
                raise ValueError()
            threshold = val
        except Exception:
            raise HTTPException(status_code=400, detail="Threshold must be a finite numeric value greater than 0.")
    else:
        threshold = version.get("calibration", {}).get("threshold", 27.0)

    artifacts = version.get("artifacts", {})
    total_files = len(file_payloads)

    for filename, file_bytes in file_payloads:
        upload_wrapper = BytesUploadFile(filename, file_bytes)
        try:
            insp_res = run_inspection(
                db=db,
                model_id=model_id,
                upload_file=upload_wrapper,
                threshold_override=threshold_override,
                inspection_mode=inspection_mode,
                min_instance_area=min_instance_area,
                max_instances=max_instances,
                run_id=run_id
            )
            completed_images += 1

            if (inspection_mode or "").lower() == "multi_instance":
                ov_status = insp_res.get("result", {}).get("overall_prediction", {}).get("status", "") or insp_res.get("overall_prediction", {}).get("status", "")
                is_reject = str(ov_status).upper() in ("REJECT", "ANOMALOUS")
            else:
                pred_status = insp_res.get("prediction", {}).get("status", "")
                is_reject = str(pred_status).upper() in ("REJECT", "ANOMALOUS")

            if is_reject:
                reject_count += 1
            else:
                pass_count += 1

            inspection_records.append(insp_res)
        except Exception as e:
            error_count += 1
            inspection_records.append({
                "filename": filename,
                "status": "failed",
                "error": str(e)
            })

        # Dynamically update progress in MongoDB after EACH image finishes
        db.inspection_runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "completed_images": completed_images,
                "pass_count": pass_count,
                "reject_count": reject_count,
                "error_count": error_count,
                "summary": {
                    "total": total_files,
                    "pass": pass_count,
                    "reject": reject_count,
                    "errors": error_count
                }
            }}
        )

    # Finalize run status
    if error_count == 0 and completed_images == total_files:
        final_status = "completed"
    elif completed_images > 0:
        final_status = "partial"
    else:
        final_status = "failed"

    completed_time = datetime.utcnow()
    db.inspection_runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {
            "status": final_status,
            "completed_images": completed_images,
            "pass_count": pass_count,
            "reject_count": reject_count,
            "error_count": error_count,
            "summary": {
                "total": total_files,
                "pass": pass_count,
                "reject": reject_count,
                "errors": error_count
            },
            "completed_at": completed_time
        }}
    )

    # Trigger Inspection Run Notification
    try:
        run_doc = db.inspection_runs.find_one({"_id": ObjectId(run_id)})
        run_number = run_doc.get("run_number", 1) if run_doc else 1
        create_run_notification(
            db,
            run_id=str(run_id),
            model_id=str(model_id),
            model_version_id=str(active_version_id),
            run_number=run_number,
            status=final_status,
            total_images=total_files,
            completed_images=completed_images,
            pass_count=pass_count,
            reject_count=reject_count,
            error_count=error_count
        )

    except Exception as n_err:
        print(f"[WARN] Failed to create inspection run notification: {n_err}")



def create_inspection_run(
    db: Database,
    model_id: str,
    upload_files: List[UploadFile],
    background_tasks: Optional[BackgroundTasks] = None,
    threshold_override: Optional[float] = None,
    inspection_mode: str = "single",
    min_instance_area: int = 500,
    max_instances: int = 20
) -> Dict[str, Any]:
    """
    Creates a batch Inspection Run for multiple test images under an ACTIVE model.
    If background_tasks is provided, starts image processing asynchronously and returns run_id immediately.
    Accepts optional threshold_override and inspection_mode parameters.
    """
    if not ObjectId.is_valid(model_id):
        raise HTTPException(status_code=400, detail=f"Invalid model_id format '{model_id}'.")

    if not upload_files or len(upload_files) == 0:
        raise HTTPException(status_code=400, detail="At least one test image file must be uploaded for an inspection run.")

    # 1. Active Model & Active Version Guard
    model = get_model(db, model_id)
    model_status = (model.get("status") or "").lower()
    if model_status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model.get('name', model_id)}' is currently in '{model_status.upper()}' state and cannot be used for inspection runs. Please activate the model first."
        )

    if not model.get("active_version_id"):
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' does not have an active model version. Build a version first.")

    active_version_id = str(model["active_version_id"])
    version = get_model_version(db, model_id, active_version_id)

    # 2. Assign chronological run_number per model
    run_number = get_next_run_number(db, model_id)

    # 3. Read image byte contents before request completes
    file_payloads = []
    for f in upload_files:
        content = f.file.read()
        filename = f.filename or "sample.png"
        file_payloads.append((filename, content))

    # 4. Create initial InspectionRun record in 'running' state
    run_doc = InspectionRunSchema(
        model_id=ObjectId(model_id),
        model_version_id=ObjectId(active_version_id),
        run_number=run_number,
        status="running",
        total_images=len(upload_files),
        completed_images=0,
        pass_count=0,
        reject_count=0,
        error_count=0,
        summary=RunSummary(total=len(upload_files), pass_count=0, reject_count=0, errors=0),
        created_at=datetime.utcnow()
    )

    run_data = run_doc.model_dump() if hasattr(run_doc, "model_dump") else run_doc.dict()
    res = db.inspection_runs.insert_one(run_data)
    run_id = str(res.inserted_id)

    # 5. Execute in background if background_tasks is supplied, or synchronously if not
    if background_tasks is not None:
        background_tasks.add_task(
            _process_inspection_run_background,
            db,
            run_id,
            model_id,
            active_version_id,
            version,
            model,
            file_payloads,
            threshold_override,
            inspection_mode,
            min_instance_area,
            max_instances
        )
    else:
        _process_inspection_run_background(
            db,
            run_id,
            model_id,
            active_version_id,
            version,
            model,
            file_payloads,
            threshold_override,
            inspection_mode,
            min_instance_area,
            max_instances
        )


    # Re-fetch updated run document
    updated_run = db.inspection_runs.find_one({"_id": ObjectId(run_id)}) or run_data
    serialized = serialize_object_ids(updated_run)
    run_resp_id = serialized.get("id") or str(serialized.get("_id", run_id))
    serialized["run_id"] = run_resp_id
    serialized["id"] = run_resp_id
    serialized["_id"] = run_resp_id
    return serialized



def get_inspection_runs(
    db: Database,
    model_id: Optional[str] = None,
    model_version_id: Optional[str] = None,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Lists inspection runs with optional filters ordered newest first."""
    query = {}
    if model_id:
        if not ObjectId.is_valid(model_id):
            raise HTTPException(status_code=400, detail="Invalid model_id format.")
        query["model_id"] = ObjectId(model_id)
    if model_version_id:
        if not ObjectId.is_valid(model_version_id):
            raise HTTPException(status_code=400, detail="Invalid model_version_id format.")
        query["model_version_id"] = ObjectId(model_version_id)
    if status:
        query["status"] = status

    cursor = db.inspection_runs.find(query).sort("created_at", -1)
    runs = []
    for doc in cursor:
        serialized = serialize_object_ids(doc)
        run_id = serialized.get("id") or str(serialized.get("_id", ""))
        serialized["run_id"] = run_id
        runs.append(serialized)
    return runs


def get_inspection_run(db: Database, run_id: str) -> Dict[str, Any]:
    """Returns detailed InspectionRun document with metadata and inspection results."""
    if not ObjectId.is_valid(run_id):
        raise HTTPException(status_code=400, detail=f"Invalid run_id format '{run_id}'.")

    doc = db.inspection_runs.find_one({"_id": ObjectId(run_id)})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Inspection run with ID '{run_id}' not found.")

    serialized_run = serialize_object_ids(doc)
    serialized_run["run_id"] = str(serialized_run.get("_id", run_id))

    # Fetch associated inspections
    cursor = db.inspections.find({"run_id": ObjectId(run_id)}).sort("created_at", -1)
    inspections = []
    for insp_doc in cursor:
        insp_id = str(insp_doc["_id"])
        res_doc = db.inspection_results.find_one({"inspection_id": ObjectId(insp_id)})
        inspections.append(normalize_inspection_payload(insp_doc, res_doc))

    serialized_run["inspections"] = inspections
    return serialized_run
