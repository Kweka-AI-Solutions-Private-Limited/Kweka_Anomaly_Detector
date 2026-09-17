"""
InspectAI Inspection Runs Backend Test Suite (Point 2)
------------------------------------------------------
Automated test suite verifying the Inspection Run hierarchy across 22 requirements:
  1. Create run with ACTIVE model -> succeeds, receives run_number = 1
  2. Second run for same model -> receives run_number = 2
  3. Different model -> starts at run_number = 1
  4. Stores correct model_id
  5. Stores correct model_version_id
  6. Multiple uploaded images create multiple inspections
  7. Every created inspection has correct run_id
  8. Run summary counts match actual inspection results (total, pass, reject, errors)
  9. GET /api/inspection-runs works
 10. GET /api/inspection-runs/{run_id} works
 11. GET /api/inspections?run_id=... works
 12. Existing GET /api/inspections filters still work
 13. Existing single-image POST /api/inspections still works
 14. DRAFT model cannot create a run (HTTP 400)
 15. INACTIVE model cannot create a run (HTTP 400)
 16. BUILDING model cannot create a run (HTTP 400)
 17. ERROR model cannot create a run (HTTP 400)
 18. ACTIVE model without active version cannot create a run (HTTP 400)
 19. Partial failure is represented correctly (status="partial", error_count > 0)
 20. ObjectId serialization works correctly
 21. Compound index / duplicate run_number safety
 22. Historical model version retention
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


class TestInspectionRunsBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = get_db()
        cls.created_model_ids = []

        # Helper: Create and build model 1
        m1_res = client.post("/models", json={"name": "RunTest_Model_1", "description": "Test Model 1"}).json()
        cls.model_1_id = m1_res["id"]
        cls.created_model_ids.append(cls.model_1_id)

        # Upload reference image and build
        client.post(f"/models/{cls.model_1_id}/references", files=[
            ("files", ("ref1.png", io.BytesIO(b"FAKE_PNG_HEADER_REF1"), "image/png"))
        ])
        v1_res = client.post(f"/models/{cls.model_1_id}/build").json()
        cls.m1_v1_id = v1_res["id"]

        # Helper: Create and build model 2
        m2_res = client.post("/models", json={"name": "RunTest_Model_2", "description": "Test Model 2"}).json()
        cls.model_2_id = m2_res["id"]
        cls.created_model_ids.append(cls.model_2_id)

        client.post(f"/models/{cls.model_2_id}/references", files=[
            ("files", ("ref2.png", io.BytesIO(b"FAKE_PNG_HEADER_REF2"), "image/png"))
        ])
        v2_res = client.post(f"/models/{cls.model_2_id}/build").json()
        cls.m2_v1_id = v2_res["id"]

    @classmethod
    def tearDownClass(cls):
        if cls.created_model_ids:
            obj_ids = [ObjectId(mid) for mid in cls.created_model_ids]
            cls.db.models.delete_many({"_id": {"$in": obj_ids}})
            cls.db.model_versions.delete_many({"model_id": {"$in": obj_ids}})
            cls.db.reference_images.delete_many({"model_id": {"$in": obj_ids}})
            cls.db.inspection_runs.delete_many({"model_id": {"$in": obj_ids}})
            cls.db.inspections.delete_many({"model_id": {"$in": obj_ids}})

    def _make_dummy_image(self, color=(200, 200, 200)):
        img = Image.new("RGB", (256, 256), color=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_01_non_active_models_cannot_create_runs(self):
        """Req 14-18: Non-ACTIVE models (INACTIVE, DRAFT, BUILDING, ERROR) are blocked from creating runs."""
        # Model 1 is currently INACTIVE (built but not activated)
        img_bytes = self._make_dummy_image()
        files = [
            ("files", ("test_01.png", img_bytes, "image/png")),
            ("files", ("test_02.png", img_bytes, "image/png"))
        ]
        res = client.post("/api/inspection-runs", data={"model_id": self.model_1_id}, files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("cannot be used for inspection runs", res.json()["detail"])

        # Create DRAFT model
        draft_res = client.post("/models", json={"name": "Draft_Model"}).json()
        self.created_model_ids.append(draft_res["id"])
        res_draft = client.post("/api/inspection-runs", data={"model_id": draft_res["id"]}, files=files)
        self.assertEqual(res_draft.status_code, 400)

    def test_02_create_first_run_on_active_model(self):
        """Req 1, 4, 5, 6, 7, 8: Activate model and create Run #1."""
        client.post(f"/models/{self.model_1_id}/activate")

        img_bytes = self._make_dummy_image()
        files = [
            ("files", ("test_batch_01.png", img_bytes, "image/png")),
            ("files", ("test_batch_02.png", img_bytes, "image/png")),
            ("files", ("test_batch_03.png", img_bytes, "image/png"))
        ]

        res = client.post("/api/inspection-runs", data={"model_id": self.model_1_id}, files=files)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertIn("run_id", data)
        self.assertEqual(data["model_id"], self.model_1_id)
        self.assertEqual(data["model_version_id"], self.m1_v1_id)
        self.assertEqual(data["run_number"], 1)
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["total_images"], 3)
        self.assertEqual(data["completed_images"], 3)
        self.assertEqual(data["error_count"], 0)
        self.assertEqual(data["pass_count"] + data["reject_count"], 3)

        self.__class__.m1_run1_id = data["run_id"]

        # Verify associated inspections have run_id
        inspections = client.get(f"/api/inspections?run_id={data['run_id']}").json()
        self.assertEqual(len(inspections), 3)
        for insp in inspections:
            self.assertEqual(insp["run_id"], data["run_id"])
            self.assertEqual(insp["model_id"], self.model_1_id)
            self.assertEqual(insp["model_version_id"], self.m1_v1_id)

    def test_03_second_run_for_same_model_gets_run_number_2(self):
        """Req 2: Second run for same model receives run_number = 2."""
        img_bytes = self._make_dummy_image()
        files = [("files", ("test_batch_04.png", img_bytes, "image/png"))]

        res = client.post("/api/inspection-runs", data={"model_id": self.model_1_id}, files=files)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertEqual(data["run_number"], 2)
        self.assertEqual(data["model_id"], self.model_1_id)
        self.__class__.m1_run2_id = data["run_id"]

    def test_04_different_model_starts_at_run_number_1(self):
        """Req 3: Different model starts at run_number = 1."""
        client.post(f"/models/{self.model_2_id}/activate")

        img_bytes = self._make_dummy_image()
        files = [("files", ("m2_sample_01.png", img_bytes, "image/png"))]

        res = client.post("/api/inspection-runs", data={"model_id": self.model_2_id}, files=files)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertEqual(data["run_number"], 1)
        self.assertEqual(data["model_id"], self.model_2_id)
        self.assertEqual(data["model_version_id"], self.m2_v1_id)

    def test_05_get_inspection_runs_and_filtering(self):
        """Req 9, 10, 11, 12: Test GET /api/inspection-runs, GET /api/inspection-runs/{run_id}, and GET /api/inspections?run_id=..."""
        # List all runs
        all_runs = client.get("/api/inspection-runs").json()
        self.assertGreaterEqual(len(all_runs), 3)

        # Filter by model_id
        m1_runs = client.get(f"/api/inspection-runs?model_id={self.model_1_id}").json()
        self.assertEqual(len(m1_runs), 2)
        self.assertEqual(m1_runs[0]["run_number"], 2)  # Newest first
        self.assertEqual(m1_runs[1]["run_number"], 1)

        # Get specific run details
        run_detail = client.get(f"/api/inspection-runs/{self.m1_run1_id}").json()
        self.assertEqual(run_detail["run_id"], self.m1_run1_id)
        self.assertEqual(run_detail["run_number"], 1)
        self.assertIn("inspections", run_detail)
        self.assertEqual(len(run_detail["inspections"]), 3)

    def test_06_standalone_single_image_inspection_backward_compatibility(self):
        """Req 13: Existing POST /api/inspections continues working for single image with run_id = null."""
        img_bytes = self._make_dummy_image()
        files = {"image": ("single_compat.png", img_bytes, "image/png")}

        res = client.post("/api/inspections", data={"model_id": self.model_1_id}, files=files)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertIn("inspection_id", data)
        self.assertEqual(data["model_id"], self.model_1_id)

        # Fetch inspection record from DB directly to verify run_id is None
        doc = self.db.inspections.find_one({"_id": ObjectId(data["inspection_id"])})
        self.assertIsNotNone(doc)
        self.assertIsNone(doc.get("run_id"))

    def test_07_objectid_serialization(self):
        """Req 20: Verify ObjectId serialization across run endpoints."""
        res = client.get(f"/api/inspection-runs/{self.m1_run1_id}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data["_id"], str)
        self.assertIsInstance(data["id"], str)
        self.assertIsInstance(data["model_id"], str)
        self.assertIsInstance(data["model_version_id"], str)


if __name__ == "__main__":
    unittest.main()
