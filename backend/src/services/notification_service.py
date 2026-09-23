"""
InspectAI Notification Service
------------------------------
Business logic for managing application completion notifications.
Handles terminal-state notifications for Model Building and Inspection Runs.
Guarantees duplicate prevention via deterministic idempotency keys and MongoDB upsert.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
from fastapi import HTTPException
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from db.schemas import NotificationSchema


def serialize_notification(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to convert MongoDB ObjectId fields to string format."""
    doc = dict(doc)
    doc["id"] = str(doc["_id"])
    doc["_id"] = str(doc["_id"])
    if doc.get("related_model_id"):
        doc["related_model_id"] = str(doc["related_model_id"])
    if doc.get("related_version_id"):
        doc["related_version_id"] = str(doc["related_version_id"])
    if doc.get("related_run_id"):
        doc["related_run_id"] = str(doc["related_run_id"])
    return doc


def create_notification(
    db: Database,
    notification_type: str,
    title: str,
    message: str,
    severity: str,
    idempotency_key: str,
    target_route: str = "/",
    related_model_id: Optional[str] = None,
    related_version_id: Optional[str] = None,
    related_run_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Creates a new notification if idempotency_key does not already exist.
    Uses MongoDB upsert to atomically guarantee exactly 1 notification per terminal event.
    Returns serialized notification dictionary.
    """
    notif_doc = {
        "type": notification_type,
        "title": title,
        "message": message,
        "severity": severity,
        "related_model_id": ObjectId(related_model_id) if related_model_id and ObjectId.is_valid(related_model_id) else None,
        "related_version_id": ObjectId(related_version_id) if related_version_id and ObjectId.is_valid(related_version_id) else None,
        "related_run_id": ObjectId(related_run_id) if related_run_id and ObjectId.is_valid(related_run_id) else None,
        "target_route": target_route,
        "read": False,
        "idempotency_key": idempotency_key,
        "created_at": datetime.utcnow()
    }

    try:
        res = db.ad_notifications.update_one(
            {"idempotency_key": idempotency_key},
            {"$setOnInsert": notif_doc},
            upsert=True
        )
        if res.upserted_id:
            doc = db.ad_notifications.find_one({"_id": res.upserted_id})
            return serialize_notification(doc)
        else:
            doc = db.ad_notifications.find_one({"idempotency_key": idempotency_key})
            return serialize_notification(doc) if doc else None
    except DuplicateKeyError:
        doc = db.ad_notifications.find_one({"idempotency_key": idempotency_key})
        return serialize_notification(doc) if doc else None


def create_model_build_notification(
    db: Database,
    model_id: str,
    version_id: str,
    version_number: int,
    status: str,  # 'ready' or 'failed'
    model_name: str,
    error_reason: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Generates a terminal Model Build notification (completed or failed)."""
    idempotency_key = f"MODEL_BUILD:{version_id}:{status}"
    target_route = f"/models/{model_id}"

    if status == "ready":
        title = f"{model_name} V{version_number} is ready"
        message = "Model building completed successfully and is ready for activation."
        severity = "success"
        n_type = "MODEL_BUILD_COMPLETED"
    else:
        title = f"{model_name} build failed"
        message = f"Model building failed: {error_reason or 'Internal error occurred during training.'}"
        severity = "error"
        n_type = "MODEL_BUILD_FAILED"

    return create_notification(
        db,
        notification_type=n_type,
        title=title,
        message=message,
        severity=severity,
        idempotency_key=idempotency_key,
        target_route=target_route,
        related_model_id=model_id,
        related_version_id=version_id
    )


def create_run_notification(
    db: Database,
    run_id: str,
    model_id: str,
    model_version_id: str,
    run_number: int,
    status: str,  # 'completed', 'partial', 'failed'
    total_images: int,
    completed_images: int,
    pass_count: int,
    reject_count: int,
    error_count: int
) -> Optional[Dict[str, Any]]:
    """Generates a terminal Inspection Run notification (completed, partial, or failed)."""
    idempotency_key = f"RUN_COMPLETE:{run_id}:{status}"
    target_route = f"/history?run_id={run_id}"

    if status == "completed":
        title = f"Inspection Run #{run_number} completed"
        message = f"{total_images} images inspected — {pass_count} PASS, {reject_count} REJECT."
        severity = "success"
        n_type = "INSPECTION_RUN_COMPLETED"
    elif status == "partial":
        title = f"Inspection Run #{run_number} partially completed"
        message = f"{completed_images} of {total_images} images processed ({error_count} errors)."
        severity = "warning"
        n_type = "INSPECTION_RUN_PARTIAL"
    else:
        title = f"Inspection Run #{run_number} failed"
        message = f"All {total_images} image inspections encountered processing errors."
        severity = "error"
        n_type = "INSPECTION_RUN_FAILED"

    return create_notification(
        db,
        notification_type=n_type,
        title=title,
        message=message,
        severity=severity,
        idempotency_key=idempotency_key,
        target_route=target_route,
        related_model_id=model_id,
        related_version_id=model_version_id,
        related_run_id=run_id
    )


def get_notifications(
    db: Database,
    limit: int = 20,
    unread_only: bool = False
) -> List[Dict[str, Any]]:
    """Lists notifications ordered newest first with optional unread filter and limit."""
    query = {}
    if unread_only:
        query["read"] = False

    cursor = db.ad_notifications.find(query).sort("created_at", -1).limit(limit)
    return [serialize_notification(doc) for doc in cursor]


def get_unread_count(db: Database) -> int:
    """Returns total count of unread notifications."""
    return db.ad_notifications.count_documents({"read": False})


def mark_notification_read(db: Database, notification_id: str) -> Dict[str, Any]:
    """Marks a single notification as read."""
    if not ObjectId.is_valid(notification_id):
        raise HTTPException(status_code=400, detail=f"Invalid notification_id format '{notification_id}'.")

    res = db.ad_notifications.update_one(
        {"_id": ObjectId(notification_id)},
        {"$set": {"read": True}}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Notification '{notification_id}' not found.")

    doc = db.ad_notifications.find_one({"_id": ObjectId(notification_id)})
    return serialize_notification(doc)


def mark_all_notifications_read(db: Database) -> Dict[str, Any]:
    """Marks all unread notifications as read."""
    res = db.ad_notifications.update_many(
        {"read": False},
        {"$set": {"read": True}}
    )
    return {
        "message": "All notifications marked as read.",
        "modified_count": res.modified_count
    }
