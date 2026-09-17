"""
InspectAI Notifications API Router
-----------------------------------
Endpoints for viewing and managing completion notifications.
Supports listing notifications, unread count queries, and marking notifications as read.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException

from db.connection import get_db
from services.notification_service import (
    get_notifications, get_unread_count, mark_notification_read, mark_all_notifications_read
)

router = APIRouter(prefix="/notifications", tags=["Notifications"])


@router.get("")
def list_notifications_endpoint(
    limit: int = Query(20, ge=1, le=100),
    unread_only: bool = Query(False),
    db=Depends(get_db)
):
    """Lists notifications ordered newest first with optional unread filter and limit."""
    return get_notifications(db, limit=limit, unread_only=unread_only)


@router.get("/unread-count")
def unread_count_endpoint(db=Depends(get_db)):
    """Returns total count of unread notifications."""
    return {"unread_count": get_unread_count(db)}


@router.patch("/read-all")
def mark_all_read_endpoint(db=Depends(get_db)):
    """Marks all unread notifications as read."""
    return mark_all_notifications_read(db)


@router.patch("/{notification_id}/read")
def mark_read_endpoint(notification_id: str, db=Depends(get_db)):
    """Marks a specific notification as read by ID."""
    return mark_notification_read(db, notification_id)
