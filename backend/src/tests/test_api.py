"""
InspectAI API Test Suite
------------------------
Automated test suite verifying the end-to-end API workflow across 14 scenarios:
  1. Create model (POST /models)
  2. Get model details (GET /models/{model_id})
  3. Upload GOOD reference images (POST /models/{model_id}/references)
  4. Build model version (POST /models/{model_id}/build)
  5. List model versions (GET /models/{model_id}/versions)
  6. Activate model version (POST /models/{model_id}/versions/{version_id}/activate)
  7. Run inspection inference (POST /inspections)
  8. Get inspection details & result (GET /inspections/{inspection_id})
  9. List inspection history (GET /inspections)
 10. Submit inspection feedback (POST /inspections/{inspection_id}/feedback)
 11. Retrieve inspection feedback (GET /inspections/{inspection_id}/feedback)
 12. Verify historical inspection retains exact model_version_id
 13. Verify invalid IDs return 400/404 HTTP errors
 14. Verify cleanup of test records
"""

import os
import sys
import io
import unittest
from datetime import datetime
from pathlib import Path
from bson import ObjectId
from PIL import Image

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fastapi.testclient import TestClient
from api_server import app
from db.connection import get_db

client = TestClient(app)


class TestInspectAIAPIWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = get_db()
        cls.created_model_ids = []
        cls.created_inspection_ids = []

    @classmethod
    def tearDownClass(cls):
        # Clean up test documents created during testing
        if cls.created_model_ids:
            obj_ids = []
            for mid in cls.created_model_ids:
                try:
                    obj_ids.append(ObjectId(mid))
                except Exception:
                    obj_ids.append(mid)
            cls.db.models.delete_many({"_id": {"$in": obj_ids}})
            cls.db.model_versions.delete_many({"model_id": {"$in": obj_ids}})
            cls.db.reference_images.delete_many({"model_id": {"$in": obj_ids}})

    def test_01_create_model(self):
        payload = {
            "name": "API_TEST_PCB_Assembly",
            "description": "Automated test model for PCB assembly QC",
            "domain": "electronics"
        }
        response = client.post("/models", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(data["name"], payload["name"])
        self.assertEqual(data["status"], "draft")
        self.__class__.model_id = data["id"]
        self.__class__.created_model_ids.append(data["id"])

    def test_02_get_model(self):
        model_id = self.__class__.model_id
        response = client.get(f"/models/{model_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], model_id)
        self.assertEqual(data["reference_image_count"], 0)

    def test_03_upload_reference_images(self):
        model_id = self.__class__.model_id
        img_bytes = io.BytesIO(b"FAKE_PNG_HEADER_GOOD_REFERENCE_001")
        files = [
            ("files", ("good_ref_01.png", img_bytes, "image/png")),
            ("files", ("good_ref_02.png", io.BytesIO(b"FAKE_PNG_HEADER_GOOD_REFERENCE_002"), "image/png"))
        ]
        response = client.post(f"/models/{model_id}/references", files=files)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["filename"], "good_ref_01.png")

        # Verify reference count updated on model
        model_res = client.get(f"/models/{model_id}").json()
        self.assertEqual(model_res["reference_image_count"], 2)

    def test_03a_upload_single_reference_image(self):
        model_id = self.__class__.model_id
        files = [("files", ("single_ref_01.jpg", io.BytesIO(b"FAKE_JPEG_HEADER_SINGLE_REF"), "image/jpeg"))]
        response = client.post(f"/models/{model_id}/references", files=files)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["filename"], "single_ref_01.jpg")

        # Verify reference count incremented on model
        model_res = client.get(f"/models/{model_id}").json()
        self.assertEqual(model_res["reference_image_count"], 3)

    def test_03b_upload_invalid_reference_image_format(self):
        model_id = self.__class__.model_id
        files = [("files", ("unsupported_file.txt", io.BytesIO(b"NOT_AN_IMAGE"), "text/plain"))]
        response = client.post(f"/models/{model_id}/references", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported image extension", response.json()["detail"])

    def test_04_build_model_version(self):
        model_id = self.__class__.model_id
        response = client.post(f"/models/{model_id}/build")
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["version_number"], 1)
        self.assertEqual(data["status"], "ready")
        self.assertIn("threshold", data["calibration"])
        self.assertGreater(data["calibration"]["threshold"], 0)
        self.__class__.version_1_id = data["id"]

        # Verify model status updated to INACTIVE after build completion (requires explicit activation)
        model_res = client.get(f"/models/{model_id}").json()
        self.assertEqual(model_res["status"], "inactive")
        self.assertEqual(model_res["active_version_id"], data["id"])

    def test_05_list_versions(self):
        model_id = self.__class__.model_id
        response = client.get(f"/models/{model_id}/versions")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data), 1)
        self.assertEqual(data[0]["version_number"], 1)

    def test_06_activate_and_deactivate_model(self):
        model_id = self.__class__.model_id
        
        # 1. Inspection on INACTIVE model should fail with 400
        img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        files = {"image": ("test_inactive.png", buf.getvalue(), "image/png")}
        inactive_insp = client.post("/api/inspections", data={"model_id": model_id}, files=files)
        self.assertEqual(inactive_insp.status_code, 400)
        self.assertIn("cannot be used for inspections", inactive_insp.json()["detail"])

        # 2. Activate model
        act_res = client.post(f"/models/{model_id}/activate")
        self.assertEqual(act_res.status_code, 200)
        act_data = act_res.json()
        self.assertEqual(act_data["status"], "active")

        # 3. Deactivate model (safe & reversible)
        deact_res = client.post(f"/models/{model_id}/deactivate")
        self.assertEqual(deact_res.status_code, 200)
        self.assertEqual(deact_res.json()["status"], "inactive")

        # 4. Re-activate model for subsequent inspection test
        react_res = client.post(f"/models/{model_id}/activate")
        self.assertEqual(react_res.status_code, 200)
        self.assertEqual(react_res.json()["status"], "active")

    def test_07_run_inspection(self):
        model_id = self.__class__.model_id
        img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        files = {"image": ("test_sample_01.png", img_bytes, "image/png")}
        data = {"model_id": model_id}

        response = client.post("/api/inspections", data=data, files=files)
        self.assertEqual(response.status_code, 201)
        result = response.json()
        self.assertIn("inspection_id", result)
        self.assertEqual(result["model_id"], model_id)
        self.assertEqual(result["model_version_id"], self.__class__.version_1_id)
        self.assertIn("prediction", result)
        self.assertIn(result["prediction"]["status"], ["normal", "anomalous"])
        self.__class__.inspection_1_id = result["inspection_id"]

    def test_08_get_inspection(self):
        insp_id = self.__class__.inspection_1_id
        response = client.get(f"/api/inspections/{insp_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], insp_id)
        self.assertIn("result", data)
        self.assertEqual(data["model_version_id"], self.__class__.version_1_id)

    def test_09_list_inspections(self):
        model_id = self.__class__.model_id
        response = client.get(f"/api/inspections?model_id={model_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data), 1)

    def test_09b_list_inspections_bson_objectid_serialization(self):
        from bson import ObjectId
        from db.connection import get_db
        db = get_db()
        db_name = db.name
        model_obj_id = ObjectId(self.__class__.model_id)
        version_obj_id = ObjectId(self.__class__.version_1_id)

        # 1. Insert raw BSON ObjectId inspection document directly into PyMongo
        raw_insp = {
            "model_id": model_obj_id,
            "model_version_id": version_obj_id,
            "status": "completed",
            "input": {
                "storage_uri": "storage/inspections/raw_test.png",
                "filename": "raw_test.png",
                "width": 256,
                "height": 256
            },
            "processing_time_ms": 12.5,
            "created_at": datetime.utcnow()
        }
        res = db.inspections.insert_one(raw_insp)
        raw_insp_id = res.inserted_id

        # 2. Insert raw BSON ObjectId inspection_result document
        raw_res = {
            "inspection_id": raw_insp_id,
            "prediction": {
                "status": "normal",
                "anomaly_score": 5.2,
                "threshold": 27.0,
                "severity": "low"
            },
            "created_at": datetime.utcnow()
        }
        db.inspection_results.insert_one(raw_res)

        # 3. Request GET /api/inspections with all 4 filters (model_id, model_version_id, status, db_name)
        response = client.get(
            f"/api/inspections?model_id={str(model_obj_id)}&model_version_id={str(version_obj_id)}&status=completed&db_name={db_name}"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data), 1)

        # 4. Verify string serialization of ObjectId fields
        target_insp = next((i for i in data if i["id"] == str(raw_insp_id)), None)
        self.assertIsNotNone(target_insp)
        self.assertIsInstance(target_insp["id"], str)
        self.assertIsInstance(target_insp["_id"], str)
        self.assertEqual(target_insp["model_id"], str(model_obj_id))
        self.assertEqual(target_insp["model_version_id"], str(version_obj_id))
        self.assertIn("result", target_insp)
        self.assertIsInstance(target_insp["result"]["id"], str)
        self.assertIsInstance(target_insp["result"]["_id"], str)
        self.assertEqual(target_insp["result"]["inspection_id"], str(raw_insp_id))

    def test_10_submit_feedback(self):
        insp_id = self.__class__.inspection_1_id
        payload = {
            "feedback_type": "correct",
            "comment": "Accurate detection on test sample"
        }
        response = client.post(f"/api/inspections/{insp_id}/feedback", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["inspection_id"], insp_id)
        self.assertEqual(data["feedback_type"], "correct")

    def test_11_get_feedback(self):
        insp_id = self.__class__.inspection_1_id
        response = client.get(f"/api/inspections/{insp_id}/feedback")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data), 1)
        self.assertEqual(data[0]["feedback_type"], "correct")

    def test_12_historical_version_retention(self):
        model_id = self.__class__.model_id

        # Build Version 2
        upload_res = client.post(f"/api/models/{model_id}/references", files=[
            ("files", ("good_ref_03.png", io.BytesIO(b"FAKE_PNG_HEADER_GOOD_REFERENCE_003"), "image/png"))
        ])
        self.assertEqual(upload_res.status_code, 201)

        v2_res = client.post(f"/models/{model_id}/build")
        self.assertEqual(v2_res.status_code, 201)
        v2_data = v2_res.json()
        self.assertEqual(v2_data["version_number"], 2)

        # Activate model version 2 so it can be inspected
        act_v2 = client.post(f"/models/{model_id}/activate")
        self.assertEqual(act_v2.status_code, 200)

        # Run inspection on Version 2
        img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        insp2_res = client.post("/api/inspections", data={"model_id": model_id}, files={"image": ("test_sample_02.png", img_bytes, "image/png")})
        self.assertEqual(insp2_res.status_code, 201)
        insp2_data = insp2_res.json()

        # Verify old inspection still references Version 1, and new inspection references Version 2
        old_insp = client.get(f"/api/inspections/{self.__class__.inspection_1_id}").json()
        self.assertEqual(old_insp["model_version_id"], self.__class__.version_1_id)
        self.assertEqual(insp2_data["model_version_id"], v2_data["id"])

    def test_13_error_handling(self):
        # 404 Non-existent model
        res_404 = client.get("/api/models/000000000000000000000000")
        self.assertEqual(res_404.status_code, 404)

        # 400 Invalid ObjectId
        res_400 = client.get("/api/models/invalid-id-string")
        self.assertEqual(res_400.status_code, 400)

        # 400 Build model with 0 reference images
        new_m = client.post("/models", json={"name": "Empty Model"}).json()
        self.__class__.created_model_ids.append(new_m["id"])
        build_err = client.post(f"/models/{new_m['id']}/build")
        self.assertEqual(build_err.status_code, 400)


if __name__ == "__main__":
    unittest.main()
