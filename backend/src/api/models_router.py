"""
InspectAI Models API Router
---------------------------
Endpoints for Model creation, listing, details, archiving,
reference image uploads, version building, and model activation/deactivation.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel

from db.connection import get_db
from services.model_service import (
    create_model, get_models, get_model, update_model, delete_model, archive_model,
    add_reference_images, get_reference_images, get_version_reference_images,
    build_model_version, get_model_versions, get_model_version, delete_model_version,
    activate_model, deactivate_model, create_custom_threshold_version, activate_model_version
)

router = APIRouter(prefix="/models", tags=["Models"])


class ModelCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    domain: Optional[str] = None
    group_id: Optional[str] = None


class ModelUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    group_id: Optional[str] = None  # Pass null/None or string ID


class ThresholdUpdateRequest(BaseModel):
    threshold: float


@router.post("", status_code=201)
def create_model_endpoint(req: ModelCreateRequest, db=Depends(get_db)):
    """Creates a new inspection model in draft status."""
    return create_model(db, name=req.name, description=req.description, domain=req.domain, group_id=req.group_id)


@router.patch("/{model_id}")
def update_model_endpoint(model_id: str, req: ModelUpdateRequest, db=Depends(get_db)):
    """Updates model metadata (name, description, domain, group_id)."""
    payload = req.model_dump(exclude_unset=True) if hasattr(req, "model_dump") else req.dict(exclude_unset=True)
    return update_model(
        db,
        model_id=model_id,
        name=payload.get("name"),
        description=payload.get("description"),
        domain=payload.get("domain"),
        group_id=payload.get("group_id", ...) if "group_id" in payload else ...
    )


@router.get("")
def list_models_endpoint(db=Depends(get_db)):
    """Lists all inspection models sorted by updated_at DESC."""
    return get_models(db)


@router.get("/{model_id}")
def get_model_endpoint(model_id: str, db=Depends(get_db)):
    """Returns details for a single model."""
    return get_model(db, model_id)


@router.delete("/{model_id}")
def delete_model_endpoint(model_id: str, db=Depends(get_db)):
    """Soft-deletes a model and all associated versions, purging disk ML artifacts."""
    return delete_model(db, model_id)


@router.delete("/{model_id}/versions/{version_id}")
def delete_version_endpoint(model_id: str, version_id: str, db=Depends(get_db)):
    """
    Deletes a specific model version and cleans up its ML checkpoint/artifacts on disk.
    Rejects deletion of the active version of an active model (HTTP 400).
    """
    return delete_model_version(db, model_id, version_id)


@router.post("/{model_id}/activate")
def activate_model_endpoint(model_id: str, db=Depends(get_db)):
    """
    Activates a built, valid model for visual inspection.
    Validates model existence, active_version_id, version readiness,
    and PatchCore checkpoint loadability before setting status to ACTIVE.
    """
    return activate_model(db, model_id)


@router.post("/{model_id}/deactivate")
def deactivate_model_endpoint(model_id: str, db=Depends(get_db)):
    """
    Deactivates an active model, disabling new inspections.
    Does NOT delete any model versions, reference images, checkpoint artifacts,
    thresholds, or inspection history. Reversible.
    """
    return deactivate_model(db, model_id)


@router.post("/{model_id}/references", status_code=201)
def upload_references_endpoint(
    model_id: str,
    files: List[UploadFile] = File(..., description="GOOD reference image files"),
    db=Depends(get_db)
):
    """Uploads GOOD reference images for a model."""
    return add_reference_images(db, model_id, files)


@router.get("/{model_id}/references")
def get_references_endpoint(model_id: str, db=Depends(get_db)):
    """Lists reference image metadata for a model."""
    return get_reference_images(db, model_id)


@router.post("/{model_id}/build", status_code=201)
def build_version_endpoint(model_id: str, db=Depends(get_db)):
    """Builds a new PatchCore model version from uploaded GOOD reference images."""
    return build_model_version(db, model_id)


@router.get("/{model_id}/versions")
def list_versions_endpoint(model_id: str, db=Depends(get_db)):
    """Lists all versions for a model, newest first."""
    return get_model_versions(db, model_id)


@router.get("/{model_id}/versions/{version_id}")
def get_version_endpoint(model_id: str, version_id: str, db=Depends(get_db)):
    """Returns complete version metadata."""
    return get_model_version(db, model_id, version_id)


@router.get("/{model_id}/versions/{version_id}/references")
@router.get("/{model_id}/versions/{version_id}/reference-images")
def get_version_references_endpoint(model_id: str, version_id: str, db=Depends(get_db)):
    """Lists exact GOOD reference images used to construct a specific model version."""
    return get_version_reference_images(db, model_id, version_id)


@router.post("/{model_id}/threshold", status_code=201)
def update_model_threshold_endpoint(
    model_id: str,
    req: ThresholdUpdateRequest,
    db=Depends(get_db)
):
    """
    Creates a new model version (N+1) with a user-specified custom anomaly threshold.
    Reuses existing PatchCore memory bank artifacts and reference images without rebuilding.
    """
    return create_custom_threshold_version(db, model_id, req.threshold)


@router.post("/{model_id}/versions/{version_id}/activate")
def activate_version_endpoint(model_id: str, version_id: str, db=Depends(get_db)):
    """Sets a specific model version as the active version for the model."""
    return activate_model_version(db, model_id, version_id)


