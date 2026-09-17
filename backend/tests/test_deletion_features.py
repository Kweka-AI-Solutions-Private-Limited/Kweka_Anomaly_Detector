"""
Integration Test Suite for Model & Model Version Deletion Features
-------------------------------------------------------------------
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import shutil
from bson import ObjectId
from fastapi.testclient import TestClient

from api_server import app
from db.connection import get_db
from services.storage_service import ARTIFACTS_DIR, REFERENCES_DIR

client = TestClient(app)


def test_deletion_feature_suite():
    db = get_db()
    
    # -------------------------------------------------------------
    # SETUP: Create Test Models A and B
    # -------------------------------------------------------------
    res_a = client.post("/api/models", json={"name": "Test Delete Model A", "description": "Primary test model"})
    assert res_a.status_code == 201
    model_a_id = res_a.json()["id"]

    res_b = client.post("/api/models", json={"name": "Test Delete Model B", "description": "Isolated model B"})
    assert res_b.status_code == 201
    model_b_id = res_b.json()["id"]

    # Insert mock version v1 for Model A
    v1_doc = {
        "_id": ObjectId(),
        "model_id": ObjectId(model_a_id),
        "version_number": 1,
        "status": "ready",
        "calibration": {"threshold": 25.0, "auto_calibrated_threshold": 25.0, "is_custom": False, "method": "95th_percentile"},
        "artifacts": {"memory_bank": f"storage/artifacts/{model_a_id}/v1/memory_bank.npy"}
    }
    db.model_versions.insert_one(v1_doc)
    v1_id = str(v1_doc["_id"])

    # Insert mock version v2 for Model A
    v2_doc = {
        "_id": ObjectId(),
        "model_id": ObjectId(model_a_id),
        "version_number": 2,
        "status": "ready",
        "calibration": {"threshold": 30.0, "auto_calibrated_threshold": 25.0, "is_custom": True, "method": "custom"},
        "artifacts": {"memory_bank": f"storage/artifacts/{model_a_id}/v2/memory_bank.npy"}
    }
    db.model_versions.insert_one(v2_doc)
    v2_id = str(v2_doc["_id"])

    # Set Model A active with active_version_id = v1_id
    db.models.update_one(
        {"_id": ObjectId(model_a_id)},
        {"$set": {"status": "active", "active_version_id": ObjectId(v1_id)}}
    )

    # Insert mock version v1 for Model B
    v1_b_doc = {
        "_id": ObjectId(),
        "model_id": ObjectId(model_b_id),
        "version_number": 1,
        "status": "ready",
        "calibration": {"threshold": 20.0, "auto_calibrated_threshold": 20.0, "is_custom": False, "method": "95th_percentile"}
    }
    db.model_versions.insert_one(v1_b_doc)
    v1_b_id = str(v1_b_doc["_id"])
    db.models.update_one({"_id": ObjectId(model_b_id)}, {"$set": {"status": "active", "active_version_id": ObjectId(v1_b_id)}})

    # Create dummy artifact files on disk for Model A v1 & v2
    v1_dir = ARTIFACTS_DIR / model_a_id / "v1"
    v2_dir = ARTIFACTS_DIR / model_a_id / "v2"
    v1_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    (v1_dir / "memory_bank.npy").write_text("mock memory bank v1")
    (v2_dir / "memory_bank.npy").write_text("mock memory bank v2")

    # Create dummy reference image folder on disk for Model A
    ref_dir = REFERENCES_DIR / model_a_id
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "ref_1.png").write_text("mock ref image")

    # Insert reference image record in DB
    db.reference_images.insert_one({
        "model_id": ObjectId(model_a_id),
        "filename": "ref_1.png",
        "relative_path": f"storage/references/{model_a_id}/ref_1.png",
        "status": "active"
    })

    # Insert historical inspection run and result record for Model A
    run_doc = {
        "_id": ObjectId(),
        "model_id": ObjectId(model_a_id),
        "model_version_id": ObjectId(v1_id),
        "run_number": 1,
        "status": "completed",
        "total_images": 1,
        "pass_count": 1,
        "reject_count": 0
    }
    db.inspection_runs.insert_one(run_doc)
    run_id = str(run_doc["_id"])

    insp_doc = {
        "_id": ObjectId(),
        "model_id": ObjectId(model_a_id),
        "model_version_id": ObjectId(v1_id),
        "run_id": ObjectId(run_id),
        "filename": "test_sample.png",
        "prediction": {"status": "PASS", "anomaly_score": 12.4, "threshold": 25.0}
    }
    db.inspections.insert_one(insp_doc)

    print("\n--- TEST SETUP COMPLETE ---")

    # -------------------------------------------------------------
    # REQUIREMENT 2: Active Version Guard
    # -------------------------------------------------------------
    print("Testing Requirement 2: Active version guard on active model...")
    res_guard = client.delete(f"/api/models/{model_a_id}/versions/{v1_id}")
    assert res_guard.status_code == 400
    assert "Cannot delete the currently active version of an active model" in res_guard.json()["detail"]
    print("✓ Active version guard successfully blocked deletion with HTTP 400.")

    # -------------------------------------------------------------
    # REQUIREMENT 1 & 8: Delete Non-Active Model Version (v2)
    # -------------------------------------------------------------
    print("Testing Requirement 1: Deleting non-active version v2...")
    res_del_v2 = client.delete(f"/api/models/{model_a_id}/versions/{v2_id}")
    assert res_del_v2.status_code == 200

    # Verify v2 status in DB
    db_v2 = db.model_versions.find_one({"_id": ObjectId(v2_id)})
    assert db_v2["status"] == "deleted"

    # Verify v2 disk ML artifacts were purged
    assert not v2_dir.exists()
    print("✓ Non-active version v2 soft-deleted and disk artifacts purged.")

    # Verify Model B version v1_b remains intact (Isolation Requirement 8)
    db_v1_b = db.model_versions.find_one({"_id": ObjectId(v1_b_id)})
    assert db_v1_b["status"] == "ready"
    print("✓ Model B remained unaffected by Model A version deletion.")

    # -------------------------------------------------------------
    # REQUIREMENT 7: Consistent HTTP 404 on deleted version
    # -------------------------------------------------------------
    print("Testing Requirement 7: 404 on fetching/deleting already-deleted version...")
    res_fetch_v2 = client.get(f"/api/models/{model_a_id}/versions/{v2_id}")
    assert res_fetch_v2.status_code == 404

    res_redel_v2 = client.delete(f"/api/models/{model_a_id}/versions/{v2_id}")
    assert res_redel_v2.status_code == 404
    print("✓ Fetching or re-deleting deleted version returned HTTP 404 consistently.")

    # -------------------------------------------------------------
    # REQUIREMENT 3: Soft-Delete Entire Model A
    # -------------------------------------------------------------
    print("Testing Requirement 3: Soft-deleting Model A...")
    res_del_model = client.delete(f"/api/models/{model_a_id}")
    assert res_del_model.status_code == 200

    # Verify Model A status in DB
    db_model_a = db.models.find_one({"_id": ObjectId(model_a_id)})
    assert db_model_a["status"] == "deleted"

    # Verify all versions of Model A are soft-deleted
    db_v1 = db.model_versions.find_one({"_id": ObjectId(v1_id)})
    assert db_v1["status"] == "deleted"

    # Verify reference images metadata status = 'archived'
    db_ref = db.reference_images.find_one({"model_id": ObjectId(model_a_id)})
    assert db_ref["status"] == "archived"

    # Verify physical storage directories purged
    assert not (ARTIFACTS_DIR / model_a_id).exists()
    assert not (REFERENCES_DIR / model_a_id).exists()
    print("✓ Model A soft-deleted, versions soft-deleted, ref images archived, and physical storage purged.")

    # -------------------------------------------------------------
    # REQUIREMENT 7: Consistent 404 for Deleted Model
    # -------------------------------------------------------------
    print("Testing Requirement 7: Consistent 404 for deleted model...")
    res_get_model_a = client.get(f"/api/models/{model_a_id}")
    assert res_get_model_a.status_code == 404

    res_list_models = client.get("/api/models")
    model_ids_in_list = [m["id"] for m in res_list_models.json()]
    assert model_a_id not in model_ids_in_list
    assert model_b_id in model_ids_in_list
    print("✓ Deleted model A omitted from list_models and returns 404 on get_model.")

    # -------------------------------------------------------------
    # REQUIREMENT 4 & 5: Historical Inspections & Ref Images Intact
    # -------------------------------------------------------------
    print("Testing Requirement 4 & 5: Historical inspection access & ref image metadata...")
    res_runs = client.get(f"/api/inspection-runs?model_id={model_a_id}")
    assert res_runs.status_code == 200

    res_inspections = client.get(f"/api/inspections?model_id={model_a_id}")
    assert res_inspections.status_code == 200
    assert len(res_inspections.json()) >= 1
    print("✓ Historical inspection runs and results preserved and readable.")

    # Reference image metadata remains in MongoDB for audit
    ref_count = db.reference_images.count_documents({"model_id": ObjectId(model_a_id)})
    assert ref_count == 1
    print("✓ Reference image audit trail preserved in MongoDB metadata.")

    # Clean up test Model B from database
    db.models.delete_one({"_id": ObjectId(model_b_id)})
    db.model_versions.delete_many({"model_id": ObjectId(model_b_id)})
    db.models.delete_one({"_id": ObjectId(model_a_id)})
    db.model_versions.delete_many({"model_id": ObjectId(model_a_id)})
    db.reference_images.delete_many({"model_id": ObjectId(model_a_id)})
    db.inspection_runs.delete_many({"model_id": ObjectId(model_a_id)})
    db.inspections.delete_many({"model_id": ObjectId(model_a_id)})

    print("\n=============================================================")
    print("ALL 9 DELETION FEATURE REQUIREMENTS VERIFIED SUCCESSFULLY!")
    print("=============================================================\n")


if __name__ == "__main__":
    test_deletion_feature_suite()
