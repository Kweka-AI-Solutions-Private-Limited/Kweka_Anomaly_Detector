"""
InspectAI MongoDB Database Initialization & Verification Script
-----------------------------------------------------------------
Executes database initialization and automated verification:
  1. Verifies MongoDB connection
  2. Programmatically creates all 6 core collections & indexes (idempotent)
  3. Tests full referential chain: model -> model_version -> reference_image -> inspection -> inspection_result -> feedback
  4. Verifies unique index constraints ((model_id, version_number) & inspection_id)
  5. Cleans up all test verification documents in a try/finally block
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from bson import ObjectId
from pymongo.errors import DuplicateKeyError, PyMongoError

# Add parent directory to sys.path if running directly
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from db.connection import get_db, close_connection, MONGODB_DATABASE
from db.indexes import ensure_indexes
from db.schemas import (
    ModelSchema, ModelVersionSchema, ReferenceImageSchema, ImageStorage,
    InspectionSchema, InspectionInput, InspectionResultSchema,
    PredictionOutput, LocalizationOutput, BoundingBox, FeedbackSchema
)


def _dump(model_obj):
    """Helper for Pydantic v1 / v2 dictionary dump compatibility."""
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


def run_db_initialization_and_test() -> bool:
    print("\n==================================================")
    print(" INSPECTAI MONGODB INITIALIZATION & VERIFICATION")
    print("==================================================")

    # 1. Connection Check
    try:
        db = get_db()
        db.command("ping")
        print(f"[OK] MongoDB connection successful! Database: '{MONGODB_DATABASE}'")
    except Exception as e:
        print(f"[ERROR] Could not connect to MongoDB database '{MONGODB_DATABASE}'!")
        print(f"        Details: {e}")
        print("\n=== CONFIGURATION REQUIRED ===")
        print("Please ensure MONGODB_URI and MONGODB_DATABASE are set in backend/.env")
        print("Example: MONGODB_URI=mongodb://localhost:27017")
        print("         MONGODB_DATABASE=inspectai")
        return False

    # 2. Idempotent Index Creation
    print("\n--- Step 1: Programmatic Index Creation ---")
    try:
        idx_results = ensure_indexes(db)
        for col_name, idx_list in idx_results.items():
            print(f"[OK] Collection '{col_name}' indexes verified ({len(idx_list)} indexes active).")
    except Exception as e:
        print(f"[ERROR] Index creation failed: {e}")
        return False

    # 3. End-to-End Referential & Unique Constraint Verification
    print("\n--- Step 2: Referential Chain & Unique Constraint Test ---")
    
    test_doc_ids = {
        "models": [],
        "model_versions": [],
        "reference_images": [],
        "inspections": [],
        "inspection_results": [],
        "feedback": []
    }

    test_success = False

    try:
        # A. Insert Test Model
        model_data = _dump(ModelSchema(
            name="TEST_Verification_PCB_Model",
            description="Automated database verification model",
            status="ready",
            model_type="patchcore",
            domain="electronics",
            reference_image_count=50
        ))
        model_res = db.models.insert_one(model_data)
        model_id = model_res.inserted_id
        test_doc_ids["models"].append(model_id)
        print(f"[OK] 1/6 Model inserted (ObjectId: {model_id})")

        # B. Insert Test Model Version
        mv_data = _dump(ModelVersionSchema(
            model_id=model_id,
            version_number=1,
            status="active"
        ))
        mv_res = db.model_versions.insert_one(mv_data)
        version_id = mv_res.inserted_id
        test_doc_ids["model_versions"].append(version_id)
        print(f"[OK] 2/6 Model Version inserted (ObjectId: {version_id}, version_number: 1)")

        # Update model's active_version_id
        db.models.update_one({"_id": model_id}, {"$set": {"active_version_id": version_id}})

        # Test Unique Constraint on (model_id, version_number)
        dup_mv_caught = False
        try:
            db.model_versions.insert_one(mv_data)
        except DuplicateKeyError:
            dup_mv_caught = True
            print("[OK] UNIQUE Constraint Verified: (model_id + version_number) prevented duplicate insert.")
        
        if not dup_mv_caught:
            raise RuntimeError("FAILED: (model_id + version_number) unique constraint did not trigger!")

        # C. Insert Test Reference Image
        ref_data = _dump(ReferenceImageSchema(
            model_id=model_id,
            version_id=version_id,
            type="good",
            storage=ImageStorage(uri="s3://inspectai-data/test_pcb_001.png"),
            filename="test_pcb_001.png",
            relative_path="train/good/test_pcb_001.png",
            width=256,
            height=256,
            file_size=102400
        ))
        ref_res = db.reference_images.insert_one(ref_data)
        ref_id = ref_res.inserted_id
        test_doc_ids["reference_images"].append(ref_id)
        print(f"[OK] 3/6 Reference Image inserted (ObjectId: {ref_id})")

        # D. Insert Test Inspection Request
        insp_data = _dump(InspectionSchema(
            model_id=model_id,
            model_version_id=version_id,
            status="completed",
            input=InspectionInput(
                storage_uri="s3://inspectai-data/sample_001.png",
                filename="sample_001.png",
                width=256,
                height=256
            ),
            processing_time_ms=48.5,
            completed_at=datetime.utcnow()
        ))
        insp_res = db.inspections.insert_one(insp_data)
        inspection_id = insp_res.inserted_id
        test_doc_ids["inspections"].append(inspection_id)
        print(f"[OK] 4/6 Inspection inserted (ObjectId: {inspection_id})")

        # E. Insert Test Inspection Result
        res_data = _dump(InspectionResultSchema(
            inspection_id=inspection_id,
            prediction=PredictionOutput(
                status="anomalous",
                anomaly_score=1.342,
                threshold=27.0,
                severity="high"
            ),
            localization=LocalizationOutput(
                bbox=BoundingBox(x=10, y=20, width=50, height=40),
                heatmap_uri="s3://inspectai-data/heatmaps/sample_001_hm.png"
            )
        ))
        res_res = db.inspection_results.insert_one(res_data)
        result_id = res_res.inserted_id
        test_doc_ids["inspection_results"].append(result_id)
        print(f"[OK] 5/6 Inspection Result inserted (ObjectId: {result_id})")

        # Test Unique Constraint on inspection_results.inspection_id
        dup_res_caught = False
        try:
            db.inspection_results.insert_one(res_data)
        except DuplicateKeyError:
            dup_res_caught = True
            print("[OK] UNIQUE Constraint Verified: inspection_results.inspection_id prevented duplicate insert.")

        if not dup_res_caught:
            raise RuntimeError("FAILED: inspection_results.inspection_id unique constraint did not trigger!")

        # F. Insert Test Feedback
        fb_data = _dump(FeedbackSchema(
            inspection_id=inspection_id,
            model_id=model_id,
            model_version_id=version_id,
            feedback_type="correct",
            comment="True anomaly on PCB component lead"
        ))
        fb_res = db.feedback.insert_one(fb_data)
        fb_id = fb_res.inserted_id
        test_doc_ids["feedback"].append(fb_id)
        print(f"[OK] 6/6 Feedback inserted (ObjectId: {fb_id})")

        # G. Verify Full Chain Traversal
        print("\n--- Step 3: Chain Traversal & Referential Consistency ---")
        fetched_fb = db.feedback.find_one({"_id": fb_id})
        fetched_insp = db.inspections.find_one({"_id": fetched_fb["inspection_id"]})
        fetched_res = db.inspection_results.find_one({"inspection_id": fetched_insp["_id"]})
        fetched_version = db.model_versions.find_one({"_id": fetched_insp["model_version_id"]})
        fetched_model = db.models.find_one({"_id": fetched_version["model_id"]})
        fetched_ref = db.reference_images.find_one({"version_id": fetched_version["_id"]})

        assert fetched_model["_id"] == model_id, "Model ID mismatch!"
        assert fetched_version["_id"] == version_id, "Version ID mismatch!"
        assert fetched_ref["_id"] == ref_id, "Reference Image ID mismatch!"
        assert fetched_insp["_id"] == inspection_id, "Inspection ID mismatch!"
        assert fetched_res["_id"] == result_id, "Inspection Result ID mismatch!"
        assert fetched_fb["_id"] == fb_id, "Feedback ID mismatch!"

        print("[OK] Full Referential Chain Validated:")
        print(f"     model ({fetched_model['name']})")
        print(f"       +-- version v{fetched_version['version_number']} ({fetched_version['status']})")
        print(f"             +-- reference_image ({fetched_ref['filename']})")
        print(f"             +-- inspection ({fetched_insp['_id']})")
        print(f"                   +-- result (score: {fetched_res['prediction']['anomaly_score']})")
        print(f"                   +-- feedback ({fetched_fb['feedback_type']})")

        test_success = True

    except Exception as e:
        print(f"[ERROR] VERIFICATION FAILED: {e}")
        test_success = False

    finally:
        # Guaranteed Teardown of Verification Documents
        print("\n--- Step 4: Verification Cleanup ---")
        cleaned_count = 0
        for col_name, ids in test_doc_ids.items():
            if ids:
                res = db[col_name].delete_many({"_id": {"$in": ids}})
                cleaned_count += res.deleted_count
        print(f"[OK] Teardown complete: Removed {cleaned_count} test verification documents from database.")

    print("\n==================================================")
    if test_success:
        print("  INSPECTAI MONGODB ARCHITECTURE TEST: PASSED")
    else:
        print("  INSPECTAI MONGODB ARCHITECTURE TEST: FAILED")
    print("==================================================\n")

    return test_success


if __name__ == "__main__":
    success = run_db_initialization_and_test()
    close_connection()
    sys.exit(0 if success else 1)
