"""
test_version_references.py — Automated Tests for Version-Specific Reference Images
-------------------------------------------------------------------------------------
Verifies:
1. Version-specific reference images can be retrieved via endpoint.
2. Version 1 reference list does not return Version 2 references.
3. Adding new reference images for Version 2 does not alter Version 1's reference set.
4. Empty/reference-unavailable cases are handled cleanly (empty list returned, no crash).
5. 404 is returned for non-existent model or version IDs.
"""

import sys
import unittest
from pathlib import Path
from bson import ObjectId

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient
from api_server import app
from db.connection import get_db
from services.model_service import create_model, get_version_reference_images

client = TestClient(app)


class TestVersionReferences(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = get_db()
        cls.created_model_ids = []

    @classmethod
    def tearDownClass(cls):
        if cls.created_model_ids:
            obj_ids = [ObjectId(mid) for mid in cls.created_model_ids]
            cls.db.models.delete_many({"_id": {"$in": obj_ids}})
            cls.db.model_versions.delete_many({"model_id": {"$in": obj_ids}})
            cls.db.reference_images.delete_many({"model_id": {"$in": obj_ids}})

    def test_version_reference_isolation(self):
        # 1. Create a model
        model = create_model(self.db, name="Version Ref Test Model")
        model_id = model["id"]
        self.created_model_ids.append(model_id)

        # 2. Insert 2 reference images for Version 1
        ref1_id = ObjectId()
        ref2_id = ObjectId()
        self.db.reference_images.insert_many([
            {
                "_id": ref1_id,
                "model_id": ObjectId(model_id),
                "version_id": None,
                "type": "good",
                "filename": "good_ref_001.png",
                "relative_path": "storage/references/m1/good_ref_001.png",
                "uploaded_at": "2026-09-08T10:00:00Z"
            },
            {
                "_id": ref2_id,
                "model_id": ObjectId(model_id),
                "version_id": None,
                "type": "good",
                "filename": "good_ref_002.png",
                "relative_path": "storage/references/m1/good_ref_002.png",
                "uploaded_at": "2026-09-08T10:01:00Z"
            }
        ])

        # 3. Create Version 1 record pointing to [ref1_id, ref2_id]
        v1_id = ObjectId()
        self.db.model_versions.insert_one({
            "_id": v1_id,
            "model_id": ObjectId(model_id),
            "version_number": 1,
            "status": "ready",
            "training": {
                "reference_count": 2,
                "reference_ids": [str(ref1_id), str(ref2_id)],
                "build_time_ms": 1500.0
            }
        })
        self.db.reference_images.update_many(
            {"_id": {"$in": [ref1_id, ref2_id]}},
            {"$set": {"version_id": v1_id}}
        )

        # 4. Insert a NEW reference image for Version 2
        ref3_id = ObjectId()
        self.db.reference_images.insert_one({
            "_id": ref3_id,
            "model_id": ObjectId(model_id),
            "version_id": None,
            "type": "good",
            "filename": "good_ref_003.png",
            "relative_path": "storage/references/m1/good_ref_003.png",
            "uploaded_at": "2026-09-09T10:00:00Z"
        })

        # 5. Create Version 2 record pointing to ALL 3 images [ref1_id, ref2_id, ref3_id]
        v2_id = ObjectId()
        self.db.model_versions.insert_one({
            "_id": v2_id,
            "model_id": ObjectId(model_id),
            "version_number": 2,
            "status": "ready",
            "training": {
                "reference_count": 3,
                "reference_ids": [str(ref1_id), str(ref2_id), str(ref3_id)],
                "build_time_ms": 2000.0
            }
        })

        # 6. Query Version 1 references via Service
        v1_refs = get_version_reference_images(self.db, model_id, str(v1_id))
        self.assertEqual(len(v1_refs), 2)
        v1_filenames = {r["filename"] for r in v1_refs}
        self.assertEqual(v1_filenames, {"good_ref_001.png", "good_ref_002.png"})
        self.assertNotIn("good_ref_003.png", v1_filenames)

        # 7. Query Version 2 references via Service
        v2_refs = get_version_reference_images(self.db, model_id, str(v2_id))
        self.assertEqual(len(v2_refs), 3)
        v2_filenames = {r["filename"] for r in v2_refs}
        self.assertEqual(v2_filenames, {"good_ref_001.png", "good_ref_002.png", "good_ref_003.png"})

        # 8. Test API endpoint GET /api/models/{model_id}/versions/{v1_id}/references
        res = client.get(f"/models/{model_id}/versions/{v1_id}/references")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 2)

    def test_empty_reference_handling(self):
        model = create_model(self.db, name="Empty Model")
        model_id = model["id"]
        self.created_model_ids.append(model_id)

        v_id = ObjectId()
        self.db.model_versions.insert_one({
            "_id": v_id,
            "model_id": ObjectId(model_id),
            "version_number": 1,
            "status": "ready",
            "training": {
                "reference_count": 0,
                "reference_ids": []
            }
        })

        res = client.get(f"/models/{model_id}/versions/{v_id}/references")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), [])

    def test_invalid_model_or_version(self):
        res = client.get(f"/models/{str(ObjectId())}/versions/{str(ObjectId())}/references")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
