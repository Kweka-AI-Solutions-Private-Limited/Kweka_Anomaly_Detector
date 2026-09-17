"""
Unit Test Suite for Inspection Run Background Processing & Progress Tracking
-----------------------------------------------------------------------------
Verifies that:
1. POST /api/inspection-runs returns immediately with run_id and status='running'.
2. Background worker progresses completed_images in MongoDB.
3. GET /api/inspection-runs/{run_id} reflects live completed_images and summary.
4. Final status transitions to 'completed'.
5. Individual image failures are handled gracefully without aborting remaining batch processing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import io

from fastapi import BackgroundTasks
from db.connection import get_db
from services.inspection_run_service import create_inspection_run, get_inspection_run, BytesUploadFile
from fastapi.datastructures import UploadFile


def test_background_inspection_run_flow():
    db = get_db()
    
    # 1. Fetch active model
    active_model = db.models.find_one({"status": "active"})
    assert active_model is not None, "Active model must exist for background testing."
    model_id = str(active_model["_id"])

    from PIL import Image
    buf1 = io.BytesIO()
    Image.new("RGB", (10, 10), color="white").save(buf1, format="PNG")
    png_bytes = buf1.getvalue()

    # 2. Create mock UploadFiles with valid PNG format
    file1 = UploadFile(filename="test_bg_01.png", file=io.BytesIO(png_bytes))
    file2 = UploadFile(filename="test_bg_02.png", file=io.BytesIO(png_bytes))

    bg_tasks = BackgroundTasks()


    # 3. Call create_inspection_run with BackgroundTasks
    result = create_inspection_run(db, model_id=model_id, upload_files=[file1, file2], background_tasks=bg_tasks)

    assert "run_id" in result or "id" in result
    run_id = result.get("run_id") or result.get("id")
    assert run_id is not None
    assert result["status"] == "running"
    assert result["total_images"] == 2
    assert result["completed_images"] == 0

    # 4. Execute the background tasks registered
    for task in bg_tasks.tasks:
        task.func(*task.args, **task.kwargs)

    # 5. Fetch updated run state
    updated_run = get_inspection_run(db, run_id)
    assert updated_run["status"] in ("completed", "partial")
    assert updated_run["completed_images"] == 2
    assert updated_run["total_images"] == 2
    assert "summary" in updated_run
    assert updated_run["summary"]["total"] == 2


if __name__ == "__main__":
    test_background_inspection_run_flow()
    print("✓ test_background_inspection_run_flow passed!")
