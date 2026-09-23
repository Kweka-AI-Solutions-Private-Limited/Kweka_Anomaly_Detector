"""
dashboard_router.py — FastAPI Router for Dashboard Summaries
------------------------------------------------------------
Exposes GET /api/dashboard/summary for operational inspection metrics.
Mounts ONLY at /api/dashboard/summary as required by Point 7B.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from db.connection import get_db
from services.dashboard_service import get_dashboard_summary

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary")
def get_dashboard_summary_endpoint(
    model_id: Optional[str] = Query(None, description="Optional filter by model ID"),
    model_version_id: Optional[str] = Query(None, description="Optional filter by model version ID"),
    start_date: Optional[str] = Query(None, description="Optional ISO start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Optional ISO end date filter (YYYY-MM-DD)"),
    db=Depends(get_db)
):
    """
    Returns structured dashboard summary metrics aggregated across inspections,
    runs, models, and feedback.
    """
    return get_dashboard_summary(
        db,
        model_id=model_id,
        model_version_id=model_version_id,
        start_date=start_date,
        end_date=end_date
    )
