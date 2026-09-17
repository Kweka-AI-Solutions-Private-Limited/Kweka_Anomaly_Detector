"""
InspectAI Model Groups API Router
---------------------------------
RESTful endpoints for model groups organizational metadata:
  - GET    /api/model-groups
  - POST   /api/model-groups
  - GET    /api/model-groups/{group_id}
  - PATCH  /api/model-groups/{group_id}
  - DELETE /api/model-groups/{group_id}
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from db.connection import get_db
from db.schemas import ModelGroupCreate, ModelGroupUpdate
from services.model_group_service import (
    create_model_group,
    get_model_groups,
    get_model_group,
    update_model_group,
    delete_model_group,
)

router = APIRouter(prefix="/model-groups", tags=["Model Groups"])


@router.get("", response_model=List[Dict[str, Any]])
def list_groups_endpoint(db: Database = Depends(get_db)):
    """Lists all model groups with non-deleted model counts."""
    return get_model_groups(db)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=Dict[str, Any])
def create_group_endpoint(payload: ModelGroupCreate, db: Database = Depends(get_db)):
    """Creates a new user-defined model group."""
    return create_model_group(db, name=payload.name, description=payload.description)


@router.get("/{group_id}", response_model=Dict[str, Any])
def get_group_endpoint(group_id: str, db: Database = Depends(get_db)):
    """Gets details of a single model group and its member models."""
    return get_model_group(db, group_id)


@router.patch("/{group_id}", response_model=Dict[str, Any])
def update_group_endpoint(group_id: str, payload: ModelGroupUpdate, db: Database = Depends(get_db)):
    """Updates model group name or description."""
    return update_model_group(db, group_id, name=payload.name, description=payload.description)


@router.delete("/{group_id}", response_model=Dict[str, Any])
def delete_group_endpoint(group_id: str, db: Database = Depends(get_db)):
    """
    Deletes a model group.
    Unassigns all member models (sets group_id=null).
    Does NOT delete any models or historical inspection data.
    """
    return delete_model_group(db, group_id)
