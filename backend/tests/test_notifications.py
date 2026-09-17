"""
Integration Test Suite for Completion Notifications System
----------------------------------------------------------
Verifies:
1. Model build completed & failed notifications.
2. Inspection run completed, partial, & failed notifications.
3. Strict idempotency / duplicate notification prevention on re-running / re-finalizing events.
4. Notification API endpoints (list, unread count, mark read, mark all read).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from api_server import app
from db.connection import get_db
from db.indexes import ensure_indexes
from services.notification_service import (
    create_model_build_notification,
    create_run_notification,
    get_notifications,
    get_unread_count,
    mark_notification_read,
    mark_all_notifications_read
)

client = TestClient(app)


def test_completion_notifications_suite():
    db = get_db()
    ensure_indexes(db)

    # Cleanup test notification records
    db.notifications.delete_many({"idempotency_key": {"$regex": "^TEST_"}})

    test_model_id = str(ObjectId())
    test_version_id = str(ObjectId())
    test_run_id = str(ObjectId())

    # -------------------------------------------------------------
    # 1. TEST MODEL BUILD COMPLETED NOTIFICATION & DEDUPLICATION
    # -------------------------------------------------------------
    print("\n--- 1. Testing Model Build Ready Notification ---")
    notif1 = create_model_build_notification(
        db,
        model_id=test_model_id,
        version_id=test_version_id,
        version_number=1,
        status="ready",
        model_name="TEST_Steel_Model"
    )
    assert notif1 is not None
    assert notif1["type"] == "MODEL_BUILD_COMPLETED"
    assert notif1["title"] == "TEST_Steel_Model V1 is ready"
    assert notif1["severity"] == "success"
    assert notif1["read"] == False
    assert notif1["target_route"] == f"/models/{test_model_id}"

    # Re-run same model build notification (duplicate attempt)
    notif1_dup = create_model_build_notification(
        db,
        model_id=test_model_id,
        version_id=test_version_id,
        version_number=1,
        status="ready",
        model_name="TEST_Steel_Model"
    )
    assert notif1_dup["id"] == notif1["id"]

    # Verify count in DB for this idempotency key is exactly 1
    count = db.notifications.count_documents({"idempotency_key": f"MODEL_BUILD:{test_version_id}:ready"})
    assert count == 1
    print("✓ Model build ready notification created & duplicate attempt ignored.")

    # -------------------------------------------------------------
    # 2. TEST MODEL BUILD FAILED NOTIFICATION & DEDUPLICATION
    # -------------------------------------------------------------
    print("\n--- 2. Testing Model Build Failed Notification ---")
    fail_version_id = str(ObjectId())
    notif2 = create_model_build_notification(
        db,
        model_id=test_model_id,
        version_id=fail_version_id,
        version_number=2,
        status="failed",
        model_name="TEST_Steel_Model",
        error_reason="Insufficient reference images."
    )
    assert notif2["type"] == "MODEL_BUILD_FAILED"
    assert "Insufficient reference images" in notif2["message"]
    assert notif2["severity"] == "error"

    # Re-trigger duplicate failure notification
    create_model_build_notification(
        db,
        model_id=test_model_id,
        version_id=fail_version_id,
        version_number=2,
        status="failed",
        model_name="TEST_Steel_Model",
        error_reason="Insufficient reference images."
    )
    assert db.notifications.count_documents({"idempotency_key": f"MODEL_BUILD:{fail_version_id}:failed"}) == 1
    print("✓ Model build failed notification created & duplicate attempt ignored.")

    # -------------------------------------------------------------
    # 3. TEST INSPECTION RUN NOTIFICATIONS (COMPLETED, PARTIAL, FAILED)
    # -------------------------------------------------------------
    print("\n--- 3. Testing Inspection Run Notifications ---")
    run_comp_id = str(ObjectId())
    notif_run1 = create_run_notification(
        db,
        run_id=run_comp_id,
        model_id=test_model_id,
        model_version_id=test_version_id,
        run_number=10,
        status="completed",
        total_images=50,
        completed_images=50,
        pass_count=45,
        reject_count=5,
        error_count=0
    )
    assert notif_run1["type"] == "INSPECTION_RUN_COMPLETED"
    assert notif_run1["severity"] == "success"
    assert "50 images inspected — 45 PASS, 5 REJECT." in notif_run1["message"]
    assert notif_run1["target_route"] == f"/history?run_id={run_comp_id}"

    # Re-finalize completed run
    create_run_notification(
        db,
        run_id=run_comp_id,
        model_id=test_model_id,
        model_version_id=test_version_id,
        run_number=10,
        status="completed",
        total_images=50,
        completed_images=50,
        pass_count=45,
        reject_count=5,
        error_count=0
    )
    assert db.notifications.count_documents({"idempotency_key": f"RUN_COMPLETE:{run_comp_id}:completed"}) == 1
    print("✓ Inspection run completed notification created & duplicate attempt ignored.")

    # Test Partial Run
    run_part_id = str(ObjectId())
    notif_part = create_run_notification(
        db,
        run_id=run_part_id,
        model_id=test_model_id,
        model_version_id=test_version_id,
        run_number=11,
        status="partial",
        total_images=10,
        completed_images=8,
        pass_count=7,
        reject_count=1,
        error_count=2
    )
    assert notif_part["type"] == "INSPECTION_RUN_PARTIAL"
    assert notif_part["severity"] == "warning"
    assert "8 of 10 images processed (2 errors)." in notif_part["message"]

    # Test Failed Run
    run_fail_id = str(ObjectId())
    notif_fail = create_run_notification(
        db,
        run_id=run_fail_id,
        model_id=test_model_id,
        model_version_id=test_version_id,
        run_number=12,
        status="failed",
        total_images=5,
        completed_images=0,
        pass_count=0,
        reject_count=0,
        error_count=5
    )
    assert notif_fail["type"] == "INSPECTION_RUN_FAILED"
    assert notif_fail["severity"] == "error"
    print("✓ Inspection run partial & failed notifications created.")

    # -------------------------------------------------------------
    # 4. API ENDPOINTS & UNREAD COUNT / READ ACTIONS
    # -------------------------------------------------------------
    print("\n--- 4. Testing Notification API Endpoints ---")
    
    # GET /api/notifications/unread-count
    res_uc = client.get("/api/notifications/unread-count")
    assert res_uc.status_code == 200
    assert res_uc.json()["unread_count"] >= 5

    # GET /api/notifications
    res_list = client.get("/api/notifications?limit=20")
    assert res_list.status_code == 200
    items = res_list.json()
    assert len(items) >= 5

    # Test repeated polling returns same notifications without creating duplicates
    res_list_poll2 = client.get("/api/notifications?limit=20")
    assert len(res_list_poll2.json()) == len(items)
    print("✓ Repeated GET /api/notifications polling produces zero duplicates.")

    # PATCH /api/notifications/{id}/read
    target_id = notif1["id"]
    res_read = client.patch(f"/api/notifications/{target_id}/read")
    assert res_read.status_code == 200
    assert res_read.json()["read"] == True

    # Verify unread count decremented by 1
    res_uc2 = client.get("/api/notifications/unread-count")
    assert res_uc2.json()["unread_count"] == res_uc.json()["unread_count"] - 1
    print("✓ Single notification marked as read.")

    # PATCH /api/notifications/read-all
    res_read_all = client.patch("/api/notifications/read-all")
    assert res_read_all.status_code == 200

    # Verify unread count is now 0
    res_uc3 = client.get("/api/notifications/unread-count")
    assert res_uc3.json()["unread_count"] == 0
    print("✓ All notifications marked as read.")

    # Cleanup created test notifications
    db.notifications.delete_many({
        "_id": {"$in": [
            ObjectId(notif1["id"]),
            ObjectId(notif2["id"]),
            ObjectId(notif_run1["id"]),
            ObjectId(notif_part["id"]),
            ObjectId(notif_fail["id"])
        ]}
    })

    print("\n=============================================================")
    print("ALL COMPLETION NOTIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=============================================================\n")


if __name__ == "__main__":
    test_completion_notifications_suite()
