"""
InspectAI Inspection Service
----------------------------
Business logic for Inspections, Results, and Feedback.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
from fastapi import HTTPException, UploadFile
from pymongo.database import Database

from db.schemas import (
    InspectionSchema, InspectionInput, InspectionResultSchema,
    PredictionOutput, LocalizationOutput, BoundingBox, FeedbackSchema,
    VLMAnalysisSchema, OverallPredictionOutput, InstanceResultSchema,
    MAX_INSTANCES, MIN_INSTANCE_AREA,
    SUPPORTED_DETECTION_FEEDBACK, SUPPORTED_VLM_FEEDBACK_CATEGORIES
)
from services.storage_service import save_inspection_image, get_storage_base_dir
from services.patchcore_service import run_patchcore_inference
from services.instance_detection_service import detect_and_crop_instances
from services.model_service import get_model, get_model_version
from services.vlm_service import analyze_inspection_evidence, get_gemini_api_key



SUPPORTED_FEEDBACK_TYPES = {
    "correct", "false_positive", "false_negative",
    "wrong_defect_type", "wrong_severity", "wrong_location", "other"
}


import math

def run_inspection(
    db: Database,
    model_id: str,
    upload_file: UploadFile,
    threshold_override: Optional[float] = None,
    inspection_mode: str = "single_image",
    min_instance_area: int = MIN_INSTANCE_AREA,
    max_instances: int = MAX_INSTANCES,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main inspection routing entrypoint.
    Routes to Pipeline A (single_image) or Pipeline B (multi_instance).
    """
    norm_mode = (inspection_mode or "").strip().lower()
    if norm_mode == "multi_instance":
        return run_multi_instance_inspection(
            db=db,
            model_id=model_id,
            upload_file=upload_file,
            threshold_override=threshold_override,
            min_instance_area=min_instance_area,
            max_instances=max_instances,
            run_id=run_id
        )
    else:
        return run_single_image_inspection(
            db=db,
            model_id=model_id,
            upload_file=upload_file,
            threshold_override=threshold_override,
            run_id=run_id
        )


def run_single_image_inspection(
    db: Database,
    model_id: str,
    upload_file: UploadFile,
    threshold_override: Optional[float] = None,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    PIPELINE A — Single Product Image Inspection.
    Evaluates one uploaded product test image using the shared PatchCore anomaly engine.
    """
    model = get_model(db, model_id)

    model_status = (model.get("status") or "").lower()
    if model_status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model.get('name', model_id)}' is currently in '{model_status.upper()}' state and cannot be used for inspections. Please activate the model first."
        )

    if not model.get("active_version_id"):
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' does not have an active model version. Build a version first.")

    active_version_id = str(model["active_version_id"])
    version = get_model_version(db, model_id, active_version_id)

    # Validate threshold_override if provided
    if threshold_override is not None:
        try:
            val = float(threshold_override)
            if math.isnan(val) or math.isinf(val) or val <= 0:
                raise ValueError()
        except Exception:
            raise HTTPException(status_code=400, detail="Threshold must be a finite numeric value greater than 0.")

    filename_str = upload_file.filename or "sample.png"

    # 1. Create preliminary inspection document (queued -> processing)
    insp_doc = InspectionSchema(
        model_id=ObjectId(model_id),
        model_version_id=ObjectId(active_version_id),
        run_id=ObjectId(run_id) if run_id and ObjectId.is_valid(run_id) else None,
        inspection_mode="single_image",
        status="processing",
        input=InspectionInput(
            storage_uri="",
            filename=filename_str,
            width=256,
            height=256
        ),
        created_at=datetime.utcnow()
    )

    data = insp_doc.model_dump() if hasattr(insp_doc, "model_dump") else insp_doc.dict()
    res = db.ad_inspections.insert_one(data)
    inspection_id = str(res.inserted_id)

    try:
        # 2. Save image using storage abstraction
        target_path, relative_uri, file_size, checksum = save_inspection_image(inspection_id, upload_file)

        # Update input storage_uri
        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {"input.storage_uri": relative_uri}}
        )

        # 3. Run shared PatchCore inference against version memory bank
        if threshold_override is not None:
            threshold = float(threshold_override)
        else:
            threshold = float(version.get("calibration", {}).get("threshold", 27.0))

        artifacts = version.get("artifacts", {})

        pred = run_patchcore_inference(target_path, artifacts, threshold)

        # 4. Create inspection_result document
        bbox_data = None
        if pred.get("bbox"):
            bbox_data = BoundingBox(
                x=pred["bbox"]["x"],
                y=pred["bbox"]["y"],
                width=pred["bbox"]["width"],
                height=pred["bbox"]["height"]
            )

        # Downstream VLM Analysis (Gemini) - DECOUPLED (User-Triggered)
        is_reject = pred["status"] in ("anomalous", "reject") or pred.get("status") == "REJECT"
        if is_reject:
            vlm_schema_obj = None
        else:
            vlm_schema_obj = VLMAnalysisSchema(
                status="skipped",
                provider="gemini",
                defect_type="Not required",
                explanation="No anomaly was detected by PatchCore."
            )

        res_doc = InspectionResultSchema(
            inspection_id=ObjectId(inspection_id),
            inspection_mode="single_image",
            prediction=PredictionOutput(
                status=pred["status"],
                anomaly_score=pred["anomaly_score"],
                threshold=pred["threshold"],
                severity=pred.get("severity", "NONE")
            ),
            localization=LocalizationOutput(
                bbox=bbox_data,
                heatmap_uri=pred.get("heatmap_uri")
            ),
            vlm_analysis=vlm_schema_obj,
            created_at=datetime.utcnow()
        )

        res_data = res_doc.model_dump() if hasattr(res_doc, "model_dump") else res_doc.dict()
        db.ad_inspections.insert_one(res_data)

        # 5. Update inspection to completed status
        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {
                "status": "completed",
                "processing_time_ms": pred.get("processing_time_ms", 0.0),
                "completed_at": datetime.utcnow()
            }}
        )

        doc = db.ad_inspections.find_one({"_id": ObjectId(inspection_id)})
        return normalize_inspection_payload(doc, res_data)

    except Exception as e:
        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {"status": "failed"}}
        )
        raise HTTPException(status_code=500, detail=f"Inspection processing failed: {str(e)}")


def run_multi_instance_inspection(
    db: Database,
    model_id: str,
    upload_file: UploadFile,
    threshold_override: Optional[float] = None,
    min_instance_area: int = MIN_INSTANCE_AREA,
    max_instances: int = MAX_INSTANCES,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entrypoint for Pipeline B (multi-product inspection).
    Checks PIPELINE_B_MODE environment variable:
    - "gemini_only" (default): Direct multi-instance inspection via Gemini VLM
    - "patchcore": Per-instance crop detection + PatchCore inference
    """
    pipeline_b_mode = os.getenv("PIPELINE_B_MODE", "gemini_only").strip().lower()
    if pipeline_b_mode == "gemini_only":
        from services.gemini_pipeline_b_service import run_gemini_only_multi_instance_inspection
        return run_gemini_only_multi_instance_inspection(
            db=db,
            model_id=model_id,
            upload_file=upload_file,
            threshold_override=threshold_override,
            min_instance_area=min_instance_area,
            max_instances=max_instances,
            run_id=run_id
        )

    import time
    start_total_time = time.time()

    model = get_model(db, model_id)
    model_status = (model.get("status") or "").lower()
    if model_status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model.get('name', model_id)}' is currently in '{model_status.upper()}' state and cannot be used for inspections."
        )

    if not model.get("active_version_id"):
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' does not have an active model version.")

    active_version_id = str(model["active_version_id"])
    version = get_model_version(db, model_id, active_version_id)

    if threshold_override is not None:
        try:
            val = float(threshold_override)
            if math.isnan(val) or math.isinf(val) or val <= 0:
                raise ValueError()
        except Exception:
            raise HTTPException(status_code=400, detail="Threshold must be a finite numeric value greater than 0.")
        threshold = val
    else:
        threshold = float(version.get("calibration", {}).get("threshold", 27.0))

    filename_str = upload_file.filename or "multi_instance_test.png"

    # 1. Create preliminary inspection document with inspection_mode="multi_instance"
    insp_doc = InspectionSchema(
        model_id=ObjectId(model_id),
        model_version_id=ObjectId(active_version_id),
        run_id=ObjectId(run_id) if run_id and ObjectId.is_valid(run_id) else None,
        inspection_mode="multi_instance",
        status="processing",
        input=InspectionInput(
            storage_uri="",
            filename=filename_str,
            width=256,
            height=256
        ),
        created_at=datetime.utcnow()
    )

    data = insp_doc.model_dump() if hasattr(insp_doc, "model_dump") else insp_doc.dict()
    res = db.ad_inspections.insert_one(data)
    inspection_id = str(res.inserted_id)

    try:
        # 2. Save original uploaded image
        target_path, relative_uri, file_size, checksum = save_inspection_image(inspection_id, upload_file)
        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {"input.storage_uri": relative_uri}}
        )

        # 3. Instance Detection & Separation Layer (Gemini VLM)
        crops_dir = target_path.parent / "crops"
        instances_detected, det_meta = detect_and_crop_instances(
            image_path=target_path,
            output_dir=crops_dir,
            min_area=min_instance_area,
            max_instances=max_instances,
            model_context=model.get("name")
        )

        det_time_ms = det_meta.get("detection_time_ms", 0.0)

        # Handle NO_PRODUCT_INSTANCES_DETECTED / NO_OBJECTS_DETECTED
        if det_meta.get("reason") in ("NO_PRODUCT_INSTANCES_DETECTED", "NO_OBJECTS_DETECTED") or (det_meta.get("status") == "completed" and not instances_detected):
            overall_pred = OverallPredictionOutput(
                status="REVIEW",
                total_instances=0,
                pass_count=0,
                reject_count=0,
                error_count=0,
                threshold=threshold,
                reason="NO_PRODUCT_INSTANCES_DETECTED",
                message=det_meta.get("message", "No physical product instances were detected in image.")
            )
            res_doc = InspectionResultSchema(
                inspection_id=ObjectId(inspection_id),
                inspection_mode="multi_instance",
                overall_prediction=overall_pred,
                instances=[],
                created_at=datetime.utcnow()
            )
            res_data = res_doc.model_dump() if hasattr(res_doc, "model_dump") else res_doc.dict()
            db.ad_inspections.insert_one(res_data)

            db.ad_inspections.update_one(
                {"_id": ObjectId(inspection_id)},
                {"$set": {"status": "review", "completed_at": datetime.utcnow()}}
            )

            doc = db.ad_inspections.find_one({"_id": ObjectId(inspection_id)})
            return normalize_inspection_payload(doc, res_data)

        # Handle GEMINI_INSTANCE_LOCALIZATION_FAILED / INSTANCE_DETECTION_FAILED
        if det_meta.get("status") == "failed" or det_meta.get("reason") in ("GEMINI_INSTANCE_LOCALIZATION_FAILED", "INSTANCE_DETECTION_FAILED"):
            overall_pred = OverallPredictionOutput(
                status="REVIEW",
                total_instances=0,
                threshold=threshold,
                reason=det_meta.get("reason", "GEMINI_INSTANCE_LOCALIZATION_FAILED"),
                message=det_meta.get("message", "Gemini instance localization failed.")
            )
            res_doc = InspectionResultSchema(
                inspection_id=ObjectId(inspection_id),
                inspection_mode="multi_instance",
                overall_prediction=overall_pred,
                instances=[],
                created_at=datetime.utcnow()
            )
            res_data = res_doc.model_dump() if hasattr(res_doc, "model_dump") else res_doc.dict()
            db.ad_inspections.insert_one(res_data)

            db.ad_inspections.update_one(
                {"_id": ObjectId(inspection_id)},
                {"$set": {"status": "failed", "completed_at": datetime.utcnow()}}
            )

            doc = db.ad_inspections.find_one({"_id": ObjectId(inspection_id)})
            return normalize_inspection_payload(doc, res_data)

        # 4. PatchCore Inference per Instance Crop
        artifacts = version.get("artifacts", {})
        instance_results: List[InstanceResultSchema] = []

        total_patchcore_time_ms = 0.0

        # Read original image dimensions for composite heatmap & bounding box percentage calculation
        orig_img_bgr = cv2.imread(str(target_path))
        if orig_img_bgr is not None:
            orig_h, orig_w = orig_img_bgr.shape[:2]
        else:
            orig_h, orig_w = 256, 256

        # Update inspection input with actual image dimensions
        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {"input.width": orig_w, "input.height": orig_h}}
        )

        composite_bgr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        storage_root = get_storage_base_dir()

        for inst in instances_detected:
            inst_id = inst["instance_id"]
            crop_path = Path(inst["crop_filepath"])
            crop_uri = inst["crop_storage_uri"]
            orig_bbox = BoundingBox(
                x=inst["bbox"]["x"],
                y=inst["bbox"]["y"],
                width=inst["bbox"]["width"],
                height=inst["bbox"]["height"]
            )
            padded_bbox = inst["padded_bbox"]

            try:
                # Runs PatchCore model independently per crop using aspect-preserving instance preprocessing
                pred = run_patchcore_inference(crop_path, artifacts, threshold, is_instance_crop=True)
                total_patchcore_time_ms += pred.get("processing_time_ms", 0.0)

                crop_heat_uri = pred.get("heatmap_uri")
                if crop_heat_uri:
                    app_root = storage_root.parent if storage_root.name == "storage" else storage_root
                    heat_path = app_root / crop_heat_uri if crop_heat_uri.startswith("storage/") else storage_root / crop_heat_uri
                    if not heat_path.exists():
                        raise FileNotFoundError(f"Instance heatmap file not found at '{heat_path}'.")
                    if heat_path.stat().st_size == 0:
                        raise ValueError(f"Instance heatmap file '{heat_path}' is 0 bytes.")

                    heat_bgr = cv2.imread(str(heat_path))
                    if heat_bgr is None:
                        raise ValueError(f"Instance heatmap file '{heat_path}' failed OpenCV image decode.")

                    # Preserve raw heatmap artifact separately for debugging/provenance
                    raw_heat_path = heat_path.parent / f"{heat_path.stem}_raw.png"
                    cv2.imwrite(str(raw_heat_path), heat_bgr)

                    # Apply Border/Background Exclusion (Visualization & Localization Layer ONLY)
                    # Preserves raw anomaly score, threshold, and PASS/REJECT status 100% UNCHANGED.
                    from services.instance_preprocessing_service import apply_border_exclusion_to_instance_heatmap_and_bbox
                    cleaned_heat_bgr, fg_bbox = apply_border_exclusion_to_instance_heatmap_and_bbox(
                        crop_path, heat_bgr, threshold
                    )

                    # Overwrite heat_path with cleaned foreground-only heatmap served to MultiInstanceViewer
                    cv2.imwrite(str(heat_path), cleaned_heat_bgr)

                    # Paste cleaned heatmap into composite heatmap in original image coordinates
                    heat_resized = cv2.resize(cleaned_heat_bgr, (padded_bbox["width"], padded_bbox["height"]))
                    composite_bgr[
                        padded_bbox["y"] : padded_bbox["y"] + padded_bbox["height"],
                        padded_bbox["x"] : padded_bbox["x"] + padded_bbox["width"]
                    ] = heat_resized

                    # Update crop_bbox to use foreground-only defect localization bbox if available
                    if fg_bbox is not None:
                        pred["bbox"] = fg_bbox

                # Coordinate Mapping: Map crop-relative anomaly bbox to original image coordinates
                mapped_bbox = None
                if pred.get("bbox"):
                    crop_bbox = pred["bbox"]
                    mapped_bbox = BoundingBox(
                        x=padded_bbox["x"] + crop_bbox["x"],
                        y=padded_bbox["y"] + crop_bbox["y"],
                        width=crop_bbox["width"],
                        height=crop_bbox["height"]
                    )

                is_reject = pred["status"] in ("anomalous", "reject") or pred.get("status") == "REJECT"
                inst_status = "REJECT" if is_reject else "PASS"

                # Decoupled VLM state per instance
                vlm_schema = None if is_reject else VLMAnalysisSchema(
                    status="skipped",
                    provider="gemini",
                    defect_type="Not required",
                    explanation="No anomaly was detected by PatchCore."
                )

                inst_obj = InstanceResultSchema(
                    instance_id=inst_id,
                    bbox=orig_bbox,
                    padded_bbox=BoundingBox(**padded_bbox),
                    crop_storage_uri=crop_uri,
                    detection_confidence=inst.get("detection_confidence", 0.95),
                    status="completed",
                    prediction=PredictionOutput(
                        status=inst_status,
                        anomaly_score=pred["anomaly_score"],
                        threshold=pred["threshold"],
                        severity=pred.get("severity", "NONE")
                    ),
                    localization=LocalizationOutput(
                        bbox=mapped_bbox,
                        heatmap_uri=pred.get("heatmap_uri")
                    ),
                    vlm_analysis=vlm_schema
                )
                instance_results.append(inst_obj)

            except Exception as inst_err:
                # Explicit error handling for individual instance failure
                instance_results.append(InstanceResultSchema(
                    instance_id=inst_id,
                    bbox=orig_bbox,
                    padded_bbox=BoundingBox(**padded_bbox),
                    crop_storage_uri=crop_uri,
                    status="failed",
                    error=str(inst_err)
                ))

        # Save & verify composite heatmap PNG
        composite_heatmap_path = target_path.parent / "composite_heatmap.png"
        cv2.imwrite(str(composite_heatmap_path), composite_bgr)

        if not composite_heatmap_path.exists() or composite_heatmap_path.stat().st_size == 0:
            raise FileNotFoundError("Composite heatmap PNG file generation failed.")

        try:
            rel_comp_uri = str(composite_heatmap_path.relative_to(storage_root)).replace("\\", "/")
        except ValueError:
            rel_comp_uri = f"storage/inspections/{inspection_id}/composite_heatmap.png"

        # 5. Overall Verdict Aggregation
        pass_cnt = sum(1 for i in instance_results if i.prediction and i.prediction.status == "PASS")
        reject_cnt = sum(1 for i in instance_results if i.prediction and i.prediction.status == "REJECT")
        err_cnt = sum(1 for i in instance_results if i.status == "failed")

        if err_cnt > 0:
            overall_status = "REVIEW"
            overall_reason = "PATCHCORE_INSTANCE_FAILED"
            overall_msg = f"Inference failed on {err_cnt} of {len(instance_results)} instance(s)."
        elif reject_cnt > 0:
            overall_status = "REJECT"
            overall_reason = None
            overall_msg = f"{reject_cnt} of {len(instance_results)} instance(s) failed inspection."
        else:
            overall_status = "PASS"
            overall_reason = None
            overall_msg = "All instances passed inspection."

        scores = [i.prediction.anomaly_score for i in instance_results if i.prediction]
        max_score = round(max(scores), 2) if scores else 0.0

        overall_pred = OverallPredictionOutput(
            status=overall_status,
            total_instances=len(instance_results),
            pass_count=pass_cnt,
            reject_count=reject_cnt,
            error_count=err_cnt,
            max_anomaly_score=max_score,
            threshold=threshold,
            reason=overall_reason,
            message=overall_msg
        )

        total_time_ms = round((time.time() - start_total_time) * 1000, 1)

        res_doc = InspectionResultSchema(
            inspection_id=ObjectId(inspection_id),
            inspection_mode="multi_instance",
            overall_prediction=overall_pred,
            instances=instance_results,
            composite_heatmap_uri=rel_comp_uri,
            processing_stats={
                "detection_time_ms": det_time_ms,
                "patchcore_time_ms": round(total_patchcore_time_ms, 1),
                "total_time_ms": total_time_ms,
                "instance_count": float(len(instance_results)),
                "image_width": orig_w,
                "image_height": orig_h
            },
            created_at=datetime.utcnow()
        )

        res_data = res_doc.model_dump() if hasattr(res_doc, "model_dump") else res_doc.dict()
        db.ad_inspections.insert_one(res_data)

        # Update inspection document status
        final_insp_status = "review" if overall_status == "REVIEW" else "completed"
        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {
                "status": final_insp_status,
                "processing_time_ms": total_time_ms,
                "completed_at": datetime.utcnow()
            }}
        )

        doc = db.ad_inspections.find_one({"_id": ObjectId(inspection_id)})
        return normalize_inspection_payload(doc, res_data)

    except Exception as e:
        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {"status": "failed"}}
        )
        raise HTTPException(status_code=500, detail=f"Multi-instance inspection processing failed: {str(e)}")


def serialize_object_ids(data: Any) -> Any:
    """
    Recursively converts BSON ObjectId instances to strings in dicts, lists, and documents.
    Preserves document structure (_id and id as string) for JSON serialization.
    """
    if isinstance(data, dict):
        res = {}
        for k, v in data.items():
            if k == "_id" and isinstance(v, ObjectId):
                res["_id"] = str(v)
                res["id"] = str(v)
            else:
                res[k] = serialize_object_ids(v)
        if "_id" in data and "id" not in res:
            res["id"] = str(data["_id"])
        return res
    elif isinstance(data, list):
        return [serialize_object_ids(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    return data


def normalize_inspection_payload(doc: Dict[str, Any], res_doc: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ensures consistent top-level fields (inspection_id, filename, storage_uri, prediction, localization)
    across all inspection service returns.
    """
    serialized = serialize_object_ids(doc)
    insp_id = serialized.get("id") or str(serialized.get("_id", ""))
    serialized["inspection_id"] = insp_id

    if "inspection_mode" in doc:
        serialized["inspection_mode"] = doc["inspection_mode"]

    # Normalize input fields
    input_data = serialized.get("input") or {}
    if not serialized.get("filename"):
        serialized["filename"] = input_data.get("filename", "")
    if not serialized.get("storage_uri"):
        serialized["storage_uri"] = input_data.get("storage_uri", "")

    # Normalize prediction & localization from result
    if res_doc:
        res_serialized = serialize_object_ids(res_doc)
        serialized["result"] = res_serialized
        if "prediction" in res_serialized and not serialized.get("prediction"):
            serialized["prediction"] = res_serialized["prediction"]
        if "localization" in res_serialized and not serialized.get("localization"):
            serialized["localization"] = res_serialized["localization"]
        if "vlm_analysis" in res_serialized:
            serialized["vlm_analysis"] = res_serialized["vlm_analysis"]
        if "overall_prediction" in res_serialized:
            serialized["overall_prediction"] = res_serialized["overall_prediction"]
        if "instances" in res_serialized:
            serialized["instances"] = res_serialized["instances"]
        if "composite_heatmap_uri" in res_serialized:
            serialized["composite_heatmap_uri"] = res_serialized["composite_heatmap_uri"]
        if "inspection_mode" in res_serialized:
            serialized["inspection_mode"] = res_serialized["inspection_mode"]

    return serialized


def get_inspections(
    db: Database,
    model_id: Optional[str] = None,
    model_version_id: Optional[str] = None,
    run_id: Optional[str] = None,
    status: Optional[str] = None,
    defect_type: Optional[str] = None,
    location: Optional[str] = None,
    severity: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    skip: int = 0,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Returns inspection history list with comprehensive filtering, date range, search, and pagination.
    Supports version isolation, run isolation, and legacy run_id=None data preservation.
    """
    query: Dict[str, Any] = {}

    if model_id:
        if not ObjectId.is_valid(model_id):
            raise HTTPException(status_code=400, detail="Invalid model_id format.")
        query["model_id"] = ObjectId(model_id)

    if model_version_id:
        if not ObjectId.is_valid(model_version_id):
            raise HTTPException(status_code=400, detail="Invalid model_version_id format.")
        query["model_version_id"] = ObjectId(model_version_id)

    if run_id:
        if run_id.lower() in ("standalone", "legacy", "null", "none"):
            query["$or"] = [{"run_id": None}, {"run_id": {"$exists": False}}]
        elif not ObjectId.is_valid(run_id):
            raise HTTPException(status_code=400, detail="Invalid run_id format.")
        else:
            query["run_id"] = ObjectId(run_id)

    # Date range filtering on created_at
    date_query = {}
    if start_date:
        try:
            s_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            date_query["$gte"] = s_dt
        except ValueError:
            pass
    if end_date:
        try:
            e_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            if len(end_date) <= 10:
                e_dt = e_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            date_query["$lte"] = e_dt
        except ValueError:
            pass
    if date_query:
        query["created_at"] = date_query

    cursor_docs = list(db.ad_inspections.find(query).sort("created_at", -1))
    inspections = []

    if not cursor_docs:
        return []

    # 1. Extract IDs for bulk pre-fetching
    insp_obj_ids = [doc["_id"] for doc in cursor_docs if "_id" in doc]
    run_obj_ids = list(set(ObjectId(doc["run_id"]) for doc in cursor_docs if doc.get("run_id") and ObjectId.is_valid(str(doc["run_id"]))))

    # 2. Bulk fetch inspection_results (support ObjectId or string inspection_id)
    results_map: Dict[str, Dict[str, Any]] = {}
    if insp_obj_ids:
        all_insp_keys = insp_obj_ids + [str(i) for i in insp_obj_ids]
        raw_results = list(db.ad_inspections.find({"inspection_id": {"$in": all_insp_keys}}))
        for r in raw_results:
            key = str(r.get("inspection_id"))
            results_map[key] = r

    # 3. Bulk fetch feedback (support ObjectId or string inspection_id)
    feedback_map: Dict[str, Dict[str, Any]] = {}
    if insp_obj_ids:
        all_insp_keys = insp_obj_ids + [str(i) for i in insp_obj_ids]
        raw_feedback = list(db.ad_feedback.find({"inspection_id": {"$in": all_insp_keys}}))
        for fb in raw_feedback:
            key = str(fb.get("inspection_id"))
            feedback_map[key] = fb

    # 4. Bulk fetch inspection_runs
    runs_map: Dict[str, Dict[str, Any]] = {}
    if run_obj_ids:
        raw_runs = list(db.ad_inspection_runs.find({"_id": {"$in": run_obj_ids}}))
        for r in raw_runs:
            key = str(r.get("_id"))
            runs_map[key] = r

    # Model map cache for name search & response metadata
    model_names: Dict[str, str] = {}
    for m in db.ad_models.find():
        model_names[str(m["_id"])] = m.get("name", "")

    for doc in cursor_docs:
        insp_id = str(doc["_id"])
        res_doc = results_map.get(insp_id)
        fb_doc = feedback_map.get(insp_id)

        if doc.get("run_id") and "run_number" not in doc:
            r_id_str = str(doc["run_id"])
            run_doc = runs_map.get(r_id_str)
            if run_doc and "run_number" in run_doc:
                doc["run_number"] = run_doc["run_number"]

        normalized = normalize_inspection_payload(doc, res_doc)

        m_id = str(doc.get("model_id", ""))
        model_name = model_names.get(m_id, "")
        normalized["model_name"] = model_name

        # Extract prediction & VLM metadata for filtering
        pred = normalized.get("prediction") or {}
        vlm = normalized.get("vlm_analysis") or {}

        pred_status = (pred.get("status") or "normal").lower()
        is_pass = pred_status in ("normal", "pass")
        is_reject = pred_status in ("anomalous", "reject")

        # 1. Status / Verdict Filter
        if status:
            st = status.lower()
            if st in ("pass", "normal") and not is_pass:
                continue
            elif st in ("reject", "anomalous") and not is_reject:
                continue

        # 2. Defect Type Filter (VLM or user feedback corrected)
        if defect_type:
            dt_req = defect_type.lower()
            act_dt = (
                (fb_doc.get("corrected_defect_type") if fb_doc else None)
                or vlm.get("defect_type")
                or ""
            ).lower()
            if dt_req not in act_dt:
                continue

        # 3. Location Filter
        if location:
            loc_req = location.lower()
            act_loc = (
                (fb_doc.get("corrected_location") if fb_doc else None)
                or vlm.get("location")
                or ""
            ).lower()
            if loc_req not in act_loc:
                continue

        # 4. Severity Filter
        if severity:
            sev_req = severity.lower()
            act_sev = (
                (fb_doc.get("corrected_severity") if fb_doc else None)
                or vlm.get("severity")
                or pred.get("severity")
                or ""
            ).lower()
            if sev_req not in act_sev:
                continue

        # 5. Search Filter (Filename or Model Name)
        if search:
            s_query = search.lower()
            fname = (normalized.get("filename") or "").lower()
            mname = model_name.lower()
            if s_query not in fname and s_query not in mname:
                continue

        inspections.append(normalized)

    # Apply pagination slice if skip/limit requested
    if skip > 0 or limit is not None:
        start_idx = skip
        end_idx = (skip + limit) if limit is not None else len(inspections)
        return inspections[start_idx:end_idx]

    return inspections



def get_inspection(db: Database, inspection_id: str) -> Dict[str, Any]:
    """Returns detailed inspection record + result."""
    if not ObjectId.is_valid(inspection_id):
        raise HTTPException(status_code=400, detail=f"Invalid inspection_id format '{inspection_id}'.")

    doc = db.ad_inspections.find_one({"_id": ObjectId(inspection_id)})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Inspection with ID '{inspection_id}' not found.")

    res_doc = db.ad_inspections.find_one({"inspection_id": ObjectId(inspection_id)})
    return normalize_inspection_payload(doc, res_doc)


def submit_feedback(
    db: Database,
    inspection_id: str,
    detection_feedback: str = "correct",
    vlm_feedback_categories: Optional[List[str]] = None,
    corrected_defect_type: Optional[str] = None,
    corrected_location: Optional[str] = None,
    corrected_severity: Optional[str] = None,
    comment: Optional[str] = None,
    feedback_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Submits structured user feedback for an inspection.
    Validates PASS vs REJECT detection feedback semantics and preserves original prediction snapshots.
    Updates existing feedback record if already submitted to prevent accidental duplicates.
    """
    insp = get_inspection(db, inspection_id)
    res_doc = db.ad_inspections.find_one({"inspection_id": ObjectId(inspection_id)})

    # Handle legacy feedback_type parameter if passed
    if feedback_type and (not detection_feedback or detection_feedback == "correct"):
        if feedback_type in ("correct", "false_positive", "false_negative"):
            detection_feedback = feedback_type
        elif feedback_type in ("wrong_defect_type", "wrong_location", "wrong_severity"):
            vlm_feedback_categories = [feedback_type]
            detection_feedback = "correct"

    vlm_cats = vlm_feedback_categories or []

    # Validate detection_feedback
    if detection_feedback not in SUPPORTED_DETECTION_FEEDBACK:
        raise HTTPException(status_code=400, detail=f"Invalid detection_feedback '{detection_feedback}'. Allowed: {SUPPORTED_DETECTION_FEEDBACK}")

    # Validate detection feedback semantics based on PatchCore status
    pred_status = (insp.get("prediction", {}).get("status") or "normal").lower()
    is_pass = pred_status in ("normal", "pass")
    is_reject = pred_status in ("anomalous", "reject")

    if is_reject and detection_feedback == "false_negative":
        raise HTTPException(status_code=400, detail="False Negative is not valid for a REJECT result. Use False Positive if PatchCore incorrectly flagged a GOOD image.")

    if is_pass and detection_feedback == "false_positive":
        raise HTTPException(status_code=400, detail="False Positive is not valid for a PASS result. Use False Negative if PatchCore missed a defect.")

    # Validate VLM feedback categories
    for cat in vlm_cats:
        if cat not in SUPPORTED_VLM_FEEDBACK_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid vlm_feedback_category '{cat}'. Allowed: {SUPPORTED_VLM_FEEDBACK_CATEGORIES}")

    # Snapshots of original predictions
    orig_pred = insp.get("prediction", {})
    orig_vlm = insp.get("vlm_analysis", {})

    # Check for existing feedback submission to update rather than duplicate
    existing = db.ad_feedback.find_one({"inspection_id": ObjectId(inspection_id)})

    fb_doc = FeedbackSchema(
        inspection_id=ObjectId(inspection_id),
        inspection_result_id=ObjectId(res_doc["_id"]) if res_doc and "_id" in res_doc else None,
        model_id=ObjectId(insp["model_id"]),
        model_version_id=ObjectId(insp["model_version_id"]),
        run_id=ObjectId(insp["run_id"]) if insp.get("run_id") and ObjectId.is_valid(insp.get("run_id")) else None,
        original_prediction=orig_pred,
        original_vlm_analysis=orig_vlm,
        detection_feedback=detection_feedback,
        vlm_feedback_categories=vlm_cats,
        corrected_defect_type=corrected_defect_type,
        corrected_location=corrected_location,
        corrected_severity=corrected_severity,
        comment=comment,
        created_at=datetime.utcnow()
    )

    data = fb_doc.model_dump() if hasattr(fb_doc, "model_dump") else fb_doc.dict()

    if existing:
        db.ad_feedback.update_one({"_id": existing["_id"]}, {"$set": data})
        data["id"] = str(existing["_id"])
        data["_id"] = str(existing["_id"])
    else:
        res = db.ad_feedback.insert_one(data)
        data["id"] = str(res.inserted_id)
        data["_id"] = str(res.inserted_id)

    data = serialize_object_ids(data)
    return data


def get_feedback(db: Database, inspection_id: str) -> List[Dict[str, Any]]:
    """Lists feedback records for a specific inspection."""
    get_inspection(db, inspection_id)  # Validate 404
    cursor = db.ad_feedback.find({"inspection_id": ObjectId(inspection_id)}).sort("created_at", -1)
    feedbacks = []
    for doc in cursor:
        feedbacks.append(serialize_object_ids(doc))
    return feedbacks


def get_all_feedback(
    db: Database,
    model_id: Optional[str] = None,
    model_version_id: Optional[str] = None,
    run_id: Optional[str] = None,
    feedback_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Returns aggregated feedback records for the Feedback Review Workspace.
    Supports filtering by model_id, model_version_id, run_id, and feedback_type.
    """
    query = {}
    if model_id and ObjectId.is_valid(model_id):
        query["model_id"] = ObjectId(model_id)
    if model_version_id and ObjectId.is_valid(model_version_id):
        query["model_version_id"] = ObjectId(model_version_id)
    if run_id and ObjectId.is_valid(run_id):
        query["run_id"] = ObjectId(run_id)

    if feedback_type:
        query["$or"] = [
            {"detection_feedback": feedback_type},
            {"vlm_feedback_categories": feedback_type},
            {"feedback_type": feedback_type}
        ]

    cursor = db.ad_feedback.find(query).sort("created_at", -1)
    feedbacks = []
    for doc in cursor:
        serialized = serialize_object_ids(doc)
        insp_id = serialized.get("inspection_id")
        if insp_id:
            try:
                insp = get_inspection(db, insp_id)
                serialized["filename"] = insp.get("filename")
                serialized["storage_uri"] = insp.get("storage_uri")
                serialized["prediction"] = insp.get("prediction")
                serialized["vlm_analysis"] = insp.get("vlm_analysis")
            except Exception:
                pass
        feedbacks.append(serialized)
    return feedbacks


def retry_vlm_analysis(db: Database, inspection_id: str, force: bool = False) -> Dict[str, Any]:
    """
    Triggers or retries Gemini VLM analysis for an existing REJECT inspection without re-running PatchCore.
    Includes backend concurrency protection ('generating' status lock) and returns cached completed analysis unless forced.
    """
    insp = get_inspection(db, inspection_id)
    insp_res_doc = db.ad_inspections.find_one({"inspection_id": ObjectId(inspection_id)})
    if not insp_res_doc:
        raise HTTPException(status_code=404, detail=f"No inspection result found for inspection '{inspection_id}'.")

    existing_vlm = insp_res_doc.get("vlm_analysis") or {}

    # If analysis is already completed and force is False, return existing persisted result without calling Gemini
    if existing_vlm.get("status") == "completed" and not force:
        return get_inspection(db, inspection_id)

    # Concurrency guard: If analysis is currently generating, reject duplicate request with 409 Conflict
    if existing_vlm.get("status") == "generating":
        raise HTTPException(status_code=409, detail="AI analysis is currently generating for this inspection.")

    # Atomically lock state to 'generating'
    db.ad_inspections.update_one(
        {"inspection_id": ObjectId(inspection_id)},
        {"$set": {
            "vlm_analysis": {
                "status": "generating",
                "provider": "gemini",
                "explanation": "AI analysis is currently generating."
            }
        }}
    )

    try:
        pred = insp.get("prediction", {})
        loc = insp.get("localization", {})
        input_info = insp.get("input", {})

        target_path = get_storage_base_dir() / input_info.get("storage_uri", "") if input_info.get("storage_uri") else None
        heatmap_path = get_storage_base_dir() / loc.get("heatmap_uri", "") if loc.get("heatmap_uri") else None

        model_doc = db.ad_models.find_one({"_id": ObjectId(insp.get("model_id"))}) if insp.get("model_id") and ObjectId.is_valid(insp.get("model_id")) else None
        model_name = model_doc.get("name") if model_doc else None

        vlm_dict = analyze_inspection_evidence(
            original_image_path=str(target_path) if target_path and target_path.exists() else "",
            heatmap_image_path=str(heatmap_path) if heatmap_path and heatmap_path.exists() else None,
            anomaly_score=pred.get("anomaly_score"),
            threshold=pred.get("threshold"),
            bbox=loc.get("bbox"),
            model_context=model_name
        )

        db.ad_inspections.update_one(
            {"inspection_id": ObjectId(inspection_id)},
            {"$set": {"vlm_analysis": vlm_dict}}
        )
    except Exception as e:
        db.ad_inspections.update_one(
            {"inspection_id": ObjectId(inspection_id)},
            {"$set": {
                "vlm_analysis": {
                    "status": "failed",
                    "provider": "gemini",
                    "explanation": f"Gemini VLM analysis encountered an error: {str(e)}"
                }
            }}
        )
        raise e

    return get_inspection(db, inspection_id)


def retry_instance_vlm_analysis(db: Database, inspection_id: str, instance_id: int, force: bool = False) -> Dict[str, Any]:
    """
    Triggers Gemini VLM analysis specifically for a REJECT instance crop.
    """
    if not ObjectId.is_valid(inspection_id):
        raise HTTPException(status_code=400, detail=f"Invalid inspection_id '{inspection_id}'.")

    insp = get_inspection(db, inspection_id)
    insp_res_doc = db.ad_inspections.find_one({"inspection_id": {"$in": [ObjectId(inspection_id), str(inspection_id)]}})
    if not insp_res_doc:
        raise HTTPException(status_code=404, detail=f"No result found for inspection '{inspection_id}'.")

    instances = insp_res_doc.get("instances") or []
    target_idx = None
    target_inst = None
    for idx, inst in enumerate(instances):
        if inst.get("instance_id") == instance_id:
            target_idx = idx
            target_inst = inst
            break

    if target_inst is None:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found in inspection '{inspection_id}'.")

    existing_vlm = target_inst.get("vlm_analysis") or {}
    if existing_vlm.get("status") == "completed" and not force:
        return get_inspection(db, inspection_id)

    # Concurrency guard for instance VLM
    if existing_vlm.get("status") == "generating":
        raise HTTPException(status_code=409, detail=f"AI analysis is currently generating for Instance {instance_id}.")

    # Lock state
    db.ad_inspections.update_one(
        {"_id": insp_res_doc["_id"], "instances.instance_id": instance_id},
        {"$set": {
            f"instances.{target_idx}.vlm_analysis": {
                "status": "generating",
                "provider": "gemini",
                "explanation": "AI analysis is currently generating."
            }
        }}
    )

    try:
        crop_rel_uri = target_inst.get("crop_storage_uri", "")
        loc = target_inst.get("localization") or {}
        heat_rel_uri = loc.get("heatmap_uri", "")
        pred = target_inst.get("prediction") or {}

        storage_root = get_storage_base_dir()
        crop_path = storage_root / crop_rel_uri if crop_rel_uri else None
        heat_path = storage_root / heat_rel_uri if heat_rel_uri else None

        model_doc = db.ad_models.find_one({"_id": ObjectId(insp.get("model_id"))}) if insp.get("model_id") and ObjectId.is_valid(insp.get("model_id")) else None
        model_name = model_doc.get("name") if model_doc else None

        vlm_dict = analyze_inspection_evidence(
            original_image_path=str(crop_path) if crop_path and crop_path.exists() else "",
            heatmap_image_path=str(heat_path) if heat_path and heat_path.exists() else None,
            anomaly_score=pred.get("anomaly_score"),
            threshold=pred.get("threshold"),
            bbox=loc.get("bbox"),
            model_context=f"{model_name} (Instance {instance_id})"
        )

        db.ad_inspections.update_one(
            {"inspection_id": ObjectId(inspection_id), "instances.instance_id": instance_id},
            {"$set": {f"instances.{target_idx}.vlm_analysis": vlm_dict}}
        )

    except Exception as e:
        db.ad_inspections.update_one(
            {"inspection_id": ObjectId(inspection_id), "instances.instance_id": instance_id},
            {"$set": {
                f"instances.{target_idx}.vlm_analysis": {
                    "status": "failed",
                    "provider": "gemini",
                    "explanation": f"Gemini VLM analysis encountered error: {str(e)}"
                }
            }}
        )
        raise e

    return get_inspection(db, inspection_id)

