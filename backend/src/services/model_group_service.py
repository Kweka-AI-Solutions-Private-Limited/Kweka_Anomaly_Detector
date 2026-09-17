"""
InspectAI Model Group Service
-----------------------------
Business logic for Model Groups organization.
Handles group creation, listing with aggregated non-deleted model counts,
updating, group member retrieval, and safe group deletion (unassigning member models).
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
from fastapi import HTTPException
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from db.schemas import ModelGroupSchema


def validate_group_name(db: Database, name: str, current_group_id: Optional[str] = None) -> str:
    """
    Validates group name presence, whitespace, maximum length (100 chars),
    and case-insensitive uniqueness against existing model_groups.
    """
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Group name is required and cannot be empty or whitespace-only.")

    trimmed = name.strip()
    if len(trimmed) > 100:
        raise HTTPException(status_code=400, detail="Group name cannot exceed 100 characters.")

    # Case-insensitive duplicate check
    regex_pattern = f"^{regex_escape(trimmed)}$"
    query: Dict[str, Any] = {"name": {"$regex": regex_pattern, "$options": "i"}}
    if current_group_id and ObjectId.is_valid(current_group_id):
        query["_id"] = {"$ne": ObjectId(current_group_id)}

    existing = db.model_groups.find_one(query)
    if existing:
        raise HTTPException(status_code=400, detail=f"A model group with the name '{trimmed}' already exists.")

    return trimmed


def regex_escape(text: str) -> str:
    """Escapes special regex characters for safe MongoDB regex matching."""
    import re
    return re.escape(text)


def create_model_group(db: Database, name: str, description: Optional[str] = None) -> Dict[str, Any]:
    """Creates a new model group."""
    trimmed_name = validate_group_name(db, name)

    doc = ModelGroupSchema(
        name=trimmed_name,
        description=description.strip() if description and description.strip() else None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    data = doc.model_dump() if hasattr(doc, "model_dump") else doc.dict()

    try:
        res = db.model_groups.insert_one(data)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail=f"A model group with the name '{trimmed_name}' already exists.")

    data["_id"] = str(res.inserted_id)
    data["id"] = str(res.inserted_id)
    data["model_count"] = 0
    return data


def get_model_groups(db: Database) -> List[Dict[str, Any]]:
    """
    Returns all model groups sorted by name ASC, with aggregated model_count
    excluding deleted models (status == 'deleted').
    """
    # 1. Single aggregation query for model counts by group_id (excludes status=='deleted')
    pipeline = [
        {"$match": {"status": {"$ne": "deleted"}, "group_id": {"$ne": None}}},
        {"$group": {"_id": "$group_id", "count": {"$sum": 1}}}
    ]
    counts_cursor = db.models.aggregate(pipeline)
    count_map = {str(item["_id"]): item["count"] for item in counts_cursor if item.get("_id")}

    # 2. Query groups
    cursor = db.model_groups.find().sort("name", 1)
    groups = []
    for doc in cursor:
        g_id = str(doc["_id"])
        doc["id"] = g_id
        doc["_id"] = g_id
        doc["model_count"] = count_map.get(g_id, 0)
        groups.append(doc)
    return groups


def get_model_group(db: Database, group_id: str) -> Dict[str, Any]:
    """Returns details for a single group including member models list."""
    if not group_id or not ObjectId.is_valid(group_id):
        raise HTTPException(status_code=400, detail=f"Invalid group_id format '{group_id}'.")

    doc = db.model_groups.find_one({"_id": ObjectId(group_id)})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Model group with ID '{group_id}' not found.")

    g_id = str(doc["_id"])
    doc["id"] = g_id
    doc["_id"] = g_id

    # Fetch member models (excluding deleted)
    models_cursor = db.models.find({"group_id": ObjectId(group_id), "status": {"$ne": "deleted"}}).sort("name", 1)
    member_models = []
    for m in models_cursor:
        m["id"] = str(m["_id"])
        m["_id"] = str(m["_id"])
        if m.get("active_version_id"):
            m["active_version_id"] = str(m["active_version_id"])
        if m.get("group_id"):
            m["group_id"] = str(m["group_id"])
        member_models.append(m)

    doc["model_count"] = len(member_models)
    doc["models"] = member_models
    return doc


def update_model_group(
    db: Database,
    group_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Updates an existing model group's name and/or description."""
    if not group_id or not ObjectId.is_valid(group_id):
        raise HTTPException(status_code=400, detail=f"Invalid group_id format '{group_id}'.")

    group = db.model_groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail=f"Model group with ID '{group_id}' not found.")

    updates: Dict[str, Any] = {"updated_at": datetime.utcnow()}

    if name is not None:
        trimmed_name = validate_group_name(db, name, current_group_id=group_id)
        updates["name"] = trimmed_name

    if description is not None:
        updates["description"] = description.strip() if description and description.strip() else None

    try:
        db.model_groups.update_one({"_id": ObjectId(group_id)}, {"$set": updates})
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="A model group with that name already exists.")

    return get_model_group(db, group_id)


def delete_model_group(db: Database, group_id: str) -> Dict[str, Any]:
    """
    Deletes a model group.
    CRITICAL: Unassigns all member models by setting group_id = None.
    NEVER deletes models, model versions, or historical inspections.
    """
    if not group_id or not ObjectId.is_valid(group_id):
        raise HTTPException(status_code=400, detail=f"Invalid group_id format '{group_id}'.")

    group = db.model_groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail=f"Model group with ID '{group_id}' not found.")

    # 1. Unassign all models belonging to this group
    db.models.update_many(
        {"group_id": ObjectId(group_id)},
        {"$set": {"group_id": None, "updated_at": datetime.utcnow()}}
    )

    # 2. Delete group document
    db.model_groups.delete_one({"_id": ObjectId(group_id)})

    return {
        "message": f"Model group '{group.get('name')}' deleted successfully. Associated models have been ungrouped.",
        "group_id": group_id
    }
