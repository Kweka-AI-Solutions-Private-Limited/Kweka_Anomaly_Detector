"""
InspectAI Model Service
-----------------------
Business logic for Models, Model Versions, and Reference Images metadata.
Supports full Model Usability Lifecycle: ACTIVE, INACTIVE, DRAFT, BUILDING, ERROR.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from bson import ObjectId
from fastapi import HTTPException, UploadFile
from pymongo.database import Database

from db.schemas import (
    ModelSchema, ModelVersionSchema, ReferenceImageSchema, ImageStorage,
    AlgorithmConfig, CalibrationConfig, TrainingConfig, VersionArtifacts
)
from services.storage_service import save_reference_image, get_version_artifacts_dir
from services.patchcore_service import (
    build_patchcore_version,
    build_pipeline_b_instance_version,
    resolve_checkpoint_path
)
from services.notification_service import create_model_build_notification



VALID_USABILITY_STATUSES = {"draft", "building", "inactive", "active", "error"}


def validate_model_artifacts_and_version(db: Database, model: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validates all criteria for activating a model:
      1. Model exists
      2. Model has active_version_id
      3. Referenced model version exists in model_versions collection
      4. Referenced version status is 'ready' or 'active'
      5. Required PatchCore checkpoint/memory-bank artifact exists on disk
      6. Artifact can be loaded successfully
    Returns: (is_valid, error_explanation)
    """
    if not model:
        return False, "Model document does not exist."

    active_v_id = model.get("active_version_id")
    if not active_v_id:
        return False, "Model does not have an active version. Build a model version first."

    if not ObjectId.is_valid(str(active_v_id)):
        return False, f"Invalid active_version_id format '{active_v_id}'."

    m_id = str(model.get("_id") or model.get("id"))
    v_doc = db.ad_models.find_one({"_id": ObjectId(str(active_v_id)), "model_id": ObjectId(m_id)})
    if not v_doc:
        return False, f"Referenced model version '{active_v_id}' was not found for model '{m_id}'."

    v_status = v_doc.get("status")
    if v_status not in ["ready", "active"]:
        return False, f"Referenced version status is '{v_status}'. Only ready versions can be activated."

    artifacts = v_doc.get("artifacts", {})
    ckpt_uri = artifacts.get("checkpoint_uri") or artifacts.get("memory_bank_uri")
    if not ckpt_uri:
        return False, "Model version metadata does not contain a checkpoint URI."

    resolved_path = resolve_checkpoint_path(ckpt_uri)
    if not resolved_path or not resolved_path.exists():
        return False, f"PatchCore checkpoint file missing from disk at '{ckpt_uri}'."

    try:
        import torch
        state_dict = torch.load(resolved_path, map_location="cpu")
        if state_dict is None:
            return False, f"Checkpoint file at '{resolved_path}' is empty or invalid."
    except Exception as e:
        return False, f"Failed to load PatchCore checkpoint from disk: {str(e)}"

    return True, None


def sync_model_usability_status(db: Database, doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrates legacy model statuses (e.g. 'ready') and enforces valid usability status.
    """
    current_status = doc.get("status", "").lower()
    
    # If already a valid new status, return
    if current_status in VALID_USABILITY_STATUSES and current_status != "ready":
        return doc

    model_id = str(doc["_id"])
    active_v_id = doc.get("active_version_id")

    if not active_v_id:
        new_status = "draft"
    else:
        is_valid, _ = validate_model_artifacts_and_version(db, doc)
        if is_valid:
            # Legacy 'ready' migrates to 'inactive' so user can explicitly activate
            new_status = "inactive"
        else:
            new_status = "error"

    db.ad_models.update_one(
        {"_id": ObjectId(model_id)},
        {"$set": {"status": new_status, "updated_at": datetime.utcnow()}}
    )
    doc["status"] = new_status
    return doc


def validate_and_parse_group_id(db: Database, group_id: Optional[str]) -> Optional[ObjectId]:
    """
    Validates group_id format and checks group existence in model_groups.
    Returns ObjectId or None.
    """
    if group_id is None:
        return None
    if isinstance(group_id, str):
        group_id_str = group_id.strip()
        if not group_id_str or group_id_str.lower() in ("null", "none", "ungrouped"):
            return None
    else:
        group_id_str = str(group_id)

    if not ObjectId.is_valid(group_id_str):
        raise HTTPException(status_code=400, detail=f"Invalid group_id format '{group_id_str}'.")

    group_obj_id = ObjectId(group_id_str)
    group_doc = db.ad_models.find_one({"_id": group_obj_id, "is_group": True})
    if not group_doc:
        raise HTTPException(status_code=404, detail=f"Model group with ID '{group_id_str}' not found.")

    return group_obj_id


def create_model(
    db: Database,
    name: str,
    description: Optional[str] = None,
    domain: Optional[str] = None,
    group_id: Optional[str] = None
) -> Dict[str, Any]:
    """Creates a new model in draft status."""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Model name is required.")

    parsed_group_id = validate_and_parse_group_id(db, group_id)

    doc = ModelSchema(
        name=name.strip(),
        description=description,
        status="draft",
        error_reason=None,
        model_type="patchcore",
        domain=domain,
        group_id=parsed_group_id,
        reference_image_count=0,
        active_version_id=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    data = doc.model_dump() if hasattr(doc, "model_dump") else doc.dict()
    res = db.ad_models.insert_one(data)
    return get_model(db, str(res.inserted_id))


def get_models(db: Database) -> List[Dict[str, Any]]:
    """Returns all models (excluding deleted) sorted by updated_at DESC."""
    cursor = list(db.ad_models.find({"status": {"$ne": "deleted"}, "is_group": {"$ne": True}}).sort("updated_at", -1))

    # Single bulk lookup for referenced model_groups to avoid N+1 queries
    group_ids = [doc["group_id"] for doc in cursor if doc.get("group_id") and ObjectId.is_valid(str(doc["group_id"]))]
    groups_map = {}
    if group_ids:
        groups_cursor = db.ad_models.find({"_id": {"$in": group_ids}, "is_group": True})
        for g in groups_cursor:
            groups_map[str(g["_id"])] = {
                "id": str(g["_id"]),
                "name": g.get("name", ""),
                "description": g.get("description")
            }

    models = []
    for doc in cursor:
        doc = sync_model_usability_status(db, doc)
        m_id = str(doc["_id"])
        doc["id"] = m_id
        doc["_id"] = m_id
        if doc.get("active_version_id"):
            doc["active_version_id"] = str(doc["active_version_id"])

        g_id = str(doc["group_id"]) if doc.get("group_id") else None
        doc["group_id"] = g_id
        doc["group"] = groups_map.get(g_id) if g_id else None
        models.append(doc)

    return models


def get_model(db: Database, model_id: str) -> Dict[str, Any]:
    """Returns model by ID or raises 404 if not found or deleted."""
    if not ObjectId.is_valid(model_id):
        raise HTTPException(status_code=400, detail=f"Invalid model_id format '{model_id}'.")

    doc = db.ad_models.find_one({"_id": ObjectId(model_id), "status": {"$ne": "deleted"}, "is_group": {"$ne": True}})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found.")

    doc = sync_model_usability_status(db, doc)
    m_id = str(doc["_id"])
    doc["id"] = m_id
    doc["_id"] = m_id
    if doc.get("active_version_id"):
        doc["active_version_id"] = str(doc["active_version_id"])

    g_id = str(doc["group_id"]) if doc.get("group_id") else None
    doc["group_id"] = g_id
    if g_id and ObjectId.is_valid(g_id):
        g_doc = db.ad_models.find_one({"_id": ObjectId(g_id)})
        doc["group"] = {
            "id": str(g_doc["_id"]),
            "name": g_doc.get("name", ""),
            "description": g_doc.get("description")
        } if g_doc else None
    else:
        doc["group"] = None

    return doc


def update_model(
    db: Database,
    model_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    domain: Optional[str] = None,
    group_id: Optional[Any] = ...  # Use Ellipsis default to distinguish omitted from explicit None
) -> Dict[str, Any]:
    """Updates model metadata (name, description, domain, group_id)."""
    model = get_model(db, model_id)  # Validates 404 if missing or deleted

    updates: Dict[str, Any] = {"updated_at": datetime.utcnow()}

    if name is not None:
        if not name or not name.strip():
            raise HTTPException(status_code=400, detail="Model name cannot be empty or whitespace-only.")
        updates["name"] = name.strip()

    if description is not None:
        updates["description"] = description.strip() if description and description.strip() else None

    if domain is not None:
        updates["domain"] = domain.strip() if domain and domain.strip() else None

    if group_id is not ...:
        parsed_group_id = validate_and_parse_group_id(db, group_id)
        updates["group_id"] = parsed_group_id

    db.ad_models.update_one({"_id": ObjectId(model_id)}, {"$set": updates})
    return get_model(db, model_id)


def activate_model(db: Database, model_id: str) -> Dict[str, Any]:
    """
    Validates model usability criteria and sets model.status = 'active'.
    Raises HTTP 400 with a detailed explanation if validation fails.
    """
    model = get_model(db, model_id)

    is_valid, error_msg = validate_model_artifacts_and_version(db, model)
    if not is_valid:
        # Update model status to error if checkpoint is missing or unloadable
        if error_msg and ("missing" in error_msg.lower() or "failed" in error_msg.lower() or "empty" in error_msg.lower()):
            db.ad_models.update_one(
                {"_id": ObjectId(model_id)},
                {"$set": {"status": "error", "error_reason": error_msg, "updated_at": datetime.utcnow()}}
            )
        raise HTTPException(status_code=400, detail=f"Cannot activate model: {error_msg}")

    db.ad_models.update_one(
        {"_id": ObjectId(model_id)},
        {"$set": {
            "status": "active",
            "error_reason": None,
            "updated_at": datetime.utcnow()
        }}
    )

    return get_model(db, model_id)


def deactivate_model(db: Database, model_id: str) -> Dict[str, Any]:
    """
    Sets model.status = 'inactive' without deleting or modifying any
    versions, references, artifacts, thresholds, or history. Safe and reversible.
    """
    model = get_model(db, model_id)

    db.ad_models.update_one(
        {"_id": ObjectId(model_id)},
        {"$set": {
            "status": "inactive",
            "updated_at": datetime.utcnow()
        }}
    )

    return get_model(db, model_id)


def add_reference_images(db: Database, model_id: str, files: List[UploadFile]) -> List[Dict[str, Any]]:
    """Uploads GOOD reference images for a model."""
    model = get_model(db, model_id)

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided for reference upload.")

    uploaded_records = []
    for file in files:
        target_path, relative_uri, file_size, checksum = save_reference_image(model_id, file)

        ref_doc = ReferenceImageSchema(
            model_id=ObjectId(model_id),
            version_id=ObjectId(model["active_version_id"]) if model.get("active_version_id") else ObjectId(),
            type="good",
            storage=ImageStorage(uri=relative_uri),
            filename=file.filename,
            relative_path=str(target_path),
            width=256,
            height=256,
            file_size=file_size,
            checksum=checksum,
            uploaded_at=datetime.utcnow()
        )

        data = ref_doc.model_dump() if hasattr(ref_doc, "model_dump") else ref_doc.dict()
        res = db.ad_models.insert_one(data)
        data["id"] = str(res.inserted_id)
        data["_id"] = str(res.inserted_id)
        data["model_id"] = str(data["model_id"])
        data["version_id"] = str(data["version_id"])
        uploaded_records.append(data)

    # Update reference image count on model
    new_count = db.ad_models.count_documents({"model_id": ObjectId(model_id)})
    db.ad_models.update_one(
        {"_id": ObjectId(model_id)},
        {"$set": {"reference_image_count": new_count, "updated_at": datetime.utcnow()}}
    )

    return uploaded_records


def get_reference_images(db: Database, model_id: str) -> List[Dict[str, Any]]:
    """Lists metadata for reference images of a model."""
    get_model(db, model_id)  # Validate 404
    cursor = db.ad_models.find({"model_id": ObjectId(model_id)}).sort("uploaded_at", -1)
    refs = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        doc["_id"] = str(doc["_id"])
        doc["model_id"] = str(doc["model_id"])
        doc["version_id"] = str(doc["version_id"])
        refs.append(doc)
    return refs


def build_model_version(db: Database, model_id: str) -> Dict[str, Any]:
    """Builds a new PatchCore model version from uploaded GOOD reference images."""
    model = get_model(db, model_id)

    # Fetch reference image docs
    ref_cursor = list(db.ad_models.find({"model_id": ObjectId(model_id)}))
    if len(ref_cursor) == 0:
        raise HTTPException(status_code=400, detail="Insufficient reference images. Please upload GOOD reference images first.")

    # Mark model status as building
    db.ad_models.update_one(
        {"_id": ObjectId(model_id)},
        {"$set": {"status": "building", "error_reason": None, "updated_at": datetime.utcnow()}}
    )

    # Determine next version number
    latest_v = db.ad_models.find_one({"model_id": ObjectId(model_id)}, sort=[("version_number", -1)])
    next_version_num = (latest_v["version_number"] + 1) if latest_v else 1

    # Create model version record in building status
    ref_ids = [str(r["_id"]) for r in ref_cursor]
    v_doc = ModelVersionSchema(
        model_id=ObjectId(model_id),
        version_number=next_version_num,
        status="building",
        algorithm=AlgorithmConfig(),
        calibration=CalibrationConfig(),
        training=TrainingConfig(reference_count=len(ref_cursor), reference_ids=ref_ids),
        artifacts=VersionArtifacts(),
        created_at=datetime.utcnow()
    )

    v_data = v_doc.model_dump() if hasattr(v_doc, "model_dump") else v_doc.dict()
    v_res = db.ad_models.insert_one(v_data)
    version_id = v_res.inserted_id

    # Gather reference image file paths
    ref_paths = [Path(r["relative_path"]) for r in ref_cursor if r.get("relative_path")]

    # Get artifacts directory
    artifacts_dir = get_version_artifacts_dir(model_id, next_version_num)

    try:
        mode = (model.get("inspection_mode") or model.get("domain") or "").lower()
        if mode == "multi_instance":
            calibrated_threshold, build_time_ms, artifact_uris = build_pipeline_b_instance_version(ref_paths, artifacts_dir)
        else:
            calibrated_threshold, build_time_ms, artifact_uris = build_patchcore_version(ref_paths, artifacts_dir)

        # Update version document to ready
        db.ad_models.update_one(
            {"_id": version_id},
            {"$set": {
                "status": "ready",
                "calibration.threshold": calibrated_threshold,
                "calibration.auto_calibrated_threshold": calibrated_threshold,
                "calibration.is_custom": False,
                "calibration.method": "95th_percentile",
                "training.build_time_ms": build_time_ms,
                "artifacts": artifact_uris
            }}
        )

        # Associate used reference images with this new version
        db.ad_models.update_many(
            {"_id": {"$in": [ObjectId(r) for r in ref_ids]}},
            {"$set": {"version_id": version_id}}
        )

        # Update model's active_version_id and set status to INACTIVE (user must explicitly activate)
        db.ad_models.update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {
                "active_version_id": version_id,
                "status": "inactive",
                "error_reason": None,
                "updated_at": datetime.utcnow()
            }}
        )

        # Trigger Model Build Completed Notification
        try:
            create_model_build_notification(
                db,
                model_id=str(model_id),
                version_id=str(version_id),
                version_number=next_version_num,
                status="ready",
                model_name=model.get("name", "Inspection Model")
            )
        except Exception as n_err:
            print(f"[WARN] Failed to create model build notification: {n_err}")

        completed_version = db.ad_models.find_one({"_id": version_id})
        completed_version["id"] = str(completed_version["_id"])
        completed_version["_id"] = str(completed_version["_id"])
        completed_version["model_id"] = str(completed_version["model_id"])
        return completed_version

    except Exception as e:
        err_msg = str(e)
        db.ad_models.update_one({"_id": version_id}, {"$set": {"status": "failed"}})
        db.ad_models.update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {"status": "error", "error_reason": err_msg, "updated_at": datetime.utcnow()}}
        )

        # Trigger Model Build Failed Notification
        try:
            create_model_build_notification(
                db,
                model_id=str(model_id),
                version_id=str(version_id),
                version_number=next_version_num,
                status="failed",
                model_name=model.get("name", "Inspection Model"),
                error_reason=err_msg
            )
        except Exception as n_err:
            print(f"[WARN] Failed to create model build failed notification: {n_err}")

        raise HTTPException(status_code=500, detail=f"PatchCore model build failed: {err_msg}")



def get_model_versions(db: Database, model_id: str) -> List[Dict[str, Any]]:
    """Returns all non-deleted versions for a model sorted newest first."""
    get_model(db, model_id)  # Validate 404
    cursor = db.ad_models.find({
        "model_id": ObjectId(model_id),
        "status": {"$ne": "deleted"}
    }).sort("version_number", -1)
    versions = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        doc["_id"] = str(doc["_id"])
        doc["model_id"] = str(doc["model_id"])
        versions.append(doc)
    return versions


def get_model_version(db: Database, model_id: str, version_id: str) -> Dict[str, Any]:
    """Returns a specific model version by ID or raises 404 if not found or deleted."""
    get_model(db, model_id)  # Validate model exists and not deleted
    if not ObjectId.is_valid(version_id):
        raise HTTPException(status_code=400, detail=f"Invalid version_id format '{version_id}'.")

    doc = db.ad_models.find_one({
        "_id": ObjectId(version_id),
        "model_id": ObjectId(model_id),
        "status": {"$ne": "deleted"}
    })
    if not doc:
        raise HTTPException(status_code=404, detail=f"Version '{version_id}' for model '{model_id}' not found.")

    doc["id"] = str(doc["_id"])
    doc["_id"] = str(doc["_id"])
    doc["model_id"] = str(doc["model_id"])
    return doc


def delete_model_version(db: Database, model_id: str, version_id: str) -> Dict[str, Any]:
    """
    Deletes a specific model version by setting status='deleted' and purging disk ML artifacts.
    Active Version Guard: Rejects deletion if version is active_version_id of an active model (HTTP 400).
    Idempotent 404: Returns HTTP 404 if model or version is missing or already deleted.
    """
    model = get_model(db, model_id)  # Raises 404 if missing or deleted
    version = get_model_version(db, model_id, version_id)  # Raises 404 if missing or deleted

    is_active_v = str(model.get("active_version_id")) == str(version_id)
    is_active_model = (model.get("status") or "").lower() == "active"

    if is_active_v and is_active_model:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the currently active version of an active model. Activate another version or deactivate the model first."
        )

    # Soft delete in DB
    db.ad_models.update_one(
        {"_id": ObjectId(version_id)},
        {"$set": {"status": "deleted", "updated_at": datetime.utcnow()}}
    )

    # If deleted version was active_version_id on an inactive model, clear active_version_id
    if is_active_v:
        # Find latest ready/inactive version to fallback to, or None
        fallback_v = db.ad_models.find_one(
            {"model_id": ObjectId(model_id), "status": {"$ne": "deleted"}},
            sort=[("version_number", -1)]
        )
        new_active_id = fallback_v["_id"] if fallback_v else None
        db.ad_models.update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {"active_version_id": new_active_id, "updated_at": datetime.utcnow()}}
        )

    # Clean up disk artifacts directory
    v_num = version.get("version_number", 1)
    artifacts_dir = get_version_artifacts_dir(model_id, v_num)
    if artifacts_dir and artifacts_dir.exists():
        import shutil
        shutil.rmtree(artifacts_dir, ignore_errors=True)

    return {
        "message": f"Model version v{v_num} deleted successfully.",
        "version_id": version_id,
        "model_id": model_id
    }


def delete_model(db: Database, model_id: str) -> Dict[str, Any]:
    """
    Soft-deletes a model and all associated versions, preserving historical inspection data.
    Purges physical storage folders for artifacts and references.
    Returns 404 consistently if model is already deleted or not found.
    """
    model = get_model(db, model_id)  # Raises 404 if missing or already deleted

    # Mark model as deleted
    db.ad_models.update_one(
        {"_id": ObjectId(model_id)},
        {"$set": {"status": "deleted", "updated_at": datetime.utcnow()}}
    )

    # Soft-delete all associated versions
    db.ad_models.update_many(
        {"model_id": ObjectId(model_id)},
        {"$set": {"status": "deleted", "updated_at": datetime.utcnow()}}
    )

    # Archive reference image records for historical auditing
    db.ad_models.update_many(
        {"model_id": ObjectId(model_id)},
        {"$set": {"status": "archived"}}
    )

    # Remove disk storage folders for model artifacts and reference images
    import shutil
    from services.storage_service import ARTIFACTS_DIR, REFERENCES_DIR

    model_art_dir = ARTIFACTS_DIR / str(model_id)
    if model_art_dir.exists():
        shutil.rmtree(model_art_dir, ignore_errors=True)

    model_ref_dir = REFERENCES_DIR / str(model_id)
    if model_ref_dir.exists():
        shutil.rmtree(model_ref_dir, ignore_errors=True)

    return {
        "message": f"Model '{model.get('name')}' and all associated versions deleted successfully.",
        "model_id": model_id
    }


def archive_model(db: Database, model_id: str) -> Dict[str, Any]:
    """Alias for delete_model to maintain backward compatibility."""
    return delete_model(db, model_id)


def activate_model_version(db: Database, model_id: str, version_id: str) -> Dict[str, Any]:
    """Activates a specific model version for the model."""
    get_model(db, model_id)
    v_doc = get_model_version(db, model_id, version_id)
    if v_doc.get("status") != "ready":
        raise HTTPException(status_code=400, detail=f"Version '{version_id}' is not in 'ready' status.")

    db.ad_models.update_one(
        {"_id": ObjectId(model_id)},
        {"$set": {
            "active_version_id": ObjectId(version_id),
            "status": "active",
            "error_reason": None,
            "updated_at": datetime.utcnow()
        }}
    )
    return get_model(db, model_id)


# Alias for backwards compatibility
activate_version = activate_model


def get_version_reference_images(db: Database, model_id: str, version_id: str) -> List[Dict[str, Any]]:
    """Lists exact reference images used to construct a specific model version."""
    get_model(db, model_id)  # Validate 404
    v_doc = get_model_version(db, model_id, version_id)

    training_cfg = v_doc.get("training", {})
    ref_ids = training_cfg.get("reference_ids", [])

    if ref_ids and len(ref_ids) > 0:
        query = {"_id": {"$in": [ObjectId(r) for r in ref_ids if ObjectId.is_valid(str(r))]}}
    else:
        # Fallback for legacy version documents
        query = {"model_id": ObjectId(model_id), "version_id": ObjectId(version_id)}

    cursor = db.ad_models.find(query).sort("uploaded_at", -1)
    refs = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        doc["_id"] = str(doc["_id"])
        doc["model_id"] = str(doc["model_id"])
        doc["version_id"] = str(doc["version_id"])
        refs.append(doc)
    return refs


def create_custom_threshold_version(db: Database, model_id: str, custom_threshold: float) -> Dict[str, Any]:
    """
    Creates a new immutable Model Version (N+1) with a user-supplied custom anomaly score threshold.
    Reuses existing checkpoint artifacts and reference image IDs without rebuilding PatchCore.
    Activates the new version for the model.
    """
    # 1. Validation: threshold must be finite positive float > 0
    if (
        custom_threshold is None or
        not isinstance(custom_threshold, (int, float)) or
        math.isnan(custom_threshold) or
        math.isinf(custom_threshold) or
        custom_threshold <= 0
    ):
        raise HTTPException(status_code=400, detail="Threshold must be a finite positive numeric value greater than 0.")

    model = get_model(db, model_id)

    # 2. Locate active version or latest valid parent version to copy detector config from
    parent_version = None
    if model.get("active_version_id") and ObjectId.is_valid(str(model["active_version_id"])):
        parent_version = db.ad_models.find_one({
            "_id": ObjectId(str(model["active_version_id"])),
            "model_id": ObjectId(model_id)
        })
    if not parent_version:
        parent_version = db.ad_models.find_one(
            {"model_id": ObjectId(model_id)},
            sort=[("version_number", -1)]
        )

    if not parent_version:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot create custom threshold version: Model '{model_id}' does not have any existing model version. Build an initial model version first."
        )

    parent_status = parent_version.get("status")
    if parent_status not in ["ready", "active"]:
        raise HTTPException(
            status_code=400,
            detail=f"Parent model version status is '{parent_status}'. A custom threshold can only be created from a ready/active model version."
        )

    # 3. Determine next version number
    latest_v = db.ad_models.find_one({"model_id": ObjectId(model_id)}, sort=[("version_number", -1)])
    next_version_num = (latest_v["version_number"] + 1) if latest_v else 1

    # 4. Determine auto_calibrated_threshold from parent version
    parent_calib = parent_version.get("calibration", {})
    auto_calib_thr = parent_calib.get("auto_calibrated_threshold")
    if auto_calib_thr is None:
        auto_calib_thr = parent_calib.get("threshold", 27.0)

    target_threshold = round(float(custom_threshold), 2)

    # 5. Construct new model version schema, copying exact algorithm, training, and artifacts
    v_doc = ModelVersionSchema(
        model_id=ObjectId(model_id),
        version_number=next_version_num,
        status="ready",
        algorithm=AlgorithmConfig(**parent_version.get("algorithm", {})),
        calibration=CalibrationConfig(
            method="custom",
            threshold=target_threshold,
            auto_calibrated_threshold=auto_calib_thr,
            is_custom=True
        ),
        training=TrainingConfig(**parent_version.get("training", {})),
        artifacts=VersionArtifacts(**parent_version.get("artifacts", {})),
        created_at=datetime.utcnow()
    )

    v_data = v_doc.model_dump() if hasattr(v_doc, "model_dump") else v_doc.dict()
    v_res = db.ad_models.insert_one(v_data)
    new_version_id = v_res.inserted_id

    # 6. Activate new version for model
    db.ad_models.update_one(
        {"_id": ObjectId(model_id)},
        {"$set": {
            "active_version_id": new_version_id,
            "status": "active",
            "error_reason": None,
            "updated_at": datetime.utcnow()
        }}
    )

    completed_version = db.ad_models.find_one({"_id": new_version_id})
    completed_version["id"] = str(completed_version["_id"])
    completed_version["_id"] = str(completed_version["_id"])
    completed_version["model_id"] = str(completed_version["model_id"])
    if "training" in completed_version and "reference_ids" in completed_version["training"]:
        completed_version["training"]["reference_ids"] = [
            str(r) for r in completed_version["training"]["reference_ids"]
        ]
    return completed_version




