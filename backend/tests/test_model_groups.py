"""
tests/test_model_groups.py
---------------------------
Integration tests for Model Groups feature using active test database.
Verifies all 20 required behaviors:
1. create group
2. list groups
3. get group
4. rename group
5. update description
6. duplicate name rejected case-insensitively
7. whitespace-only name rejected
8. invalid group_id rejected
9. nonexistent group rejected
10. create model with group
11. assign existing model to group
12. move model from group A to B
13. ungroup model
14. delete group
15. deleting group leaves models intact
16. deleted group's models become group_id=null
17. deleted models excluded from group counts
18. historical inspection remains unchanged
19. deleted model cannot be assigned
20. database state consistency
"""

import sys
from pathlib import Path
import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from api_server import app
from db.connection import get_db
from db.indexes import ensure_indexes
from services.model_group_service import (
    create_model_group, get_model_groups, get_model_group,
    update_model_group, delete_model_group
)
from services.model_service import (
    create_model, get_models, get_model, update_model, delete_model
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_teardown_db():
    """Cleans up test records created during test execution."""
    db = get_db()
    ensure_indexes(db)
    # Delete test groups and test models from ad_models
    db.ad_models.delete_many({"name": {"$regex": "^TEST_MG_"}})
    yield
    db.ad_models.delete_many({"name": {"$regex": "^TEST_MG_"}})


def test_model_groups_full_suite():
    db = get_db()

    # 1. Create Group
    g1 = create_model_group(db, name="TEST_MG_Electronics", description="PCB inspection group")
    assert g1["name"] == "TEST_MG_Electronics"
    assert g1["description"] == "PCB inspection group"
    assert "id" in g1
    assert g1["model_count"] == 0

    # 2. List Groups
    g2 = create_model_group(db, name="TEST_MG_Construction")
    all_groups = get_model_groups(db)
    group_names = [g["name"] for g in all_groups]
    assert "TEST_MG_Electronics" in group_names
    assert "TEST_MG_Construction" in group_names

    # 3. Get Group
    g_fetched = get_model_group(db, g1["id"])
    assert g_fetched["id"] == g1["id"]
    assert g_fetched["name"] == "TEST_MG_Electronics"
    assert g_fetched["model_count"] == 0

    # 4 & 5. Rename Group & Update Description
    g1_updated = update_model_group(db, g1["id"], name="TEST_MG_Electronics_Renamed", description="Updated description")
    assert g1_updated["name"] == "TEST_MG_Electronics_Renamed"
    assert g1_updated["description"] == "Updated description"

    # 6. Duplicate Name Rejected Case-Insensitively
    res_dup = client.post("/api/model-groups", json={"name": "test_mg_electronics_renamed"})
    assert res_dup.status_code == 400
    assert "already exists" in res_dup.json()["detail"].lower()

    # 7. Whitespace-Only Name Rejected
    res_ws = client.post("/api/model-groups", json={"name": "    "})
    assert res_ws.status_code == 400
    assert "required" in res_ws.json()["detail"].lower()

    # 8. Invalid group_id Rejected
    res_inv = client.get("/api/model-groups/invalid-id-xyz")
    assert res_inv.status_code == 400

    # 9. Nonexistent group_id Rejected
    fake_id = str(ObjectId())
    res_404 = client.get(f"/api/model-groups/{fake_id}")
    assert res_404.status_code == 404

    # 10. Create Model with Group
    res_m1 = client.post("/api/models", json={"name": "TEST_MG_Model1", "group_id": g1["id"]})
    assert res_m1.status_code == 201
    m1 = res_m1.json()
    assert m1["group_id"] == g1["id"]
    assert m1["group"]["name"] == "TEST_MG_Electronics_Renamed"

    # 11. Assign Existing Model to Group
    m2 = create_model(db, name="TEST_MG_Model2")
    assert m2["group_id"] is None
    m2_assigned = update_model(db, m2["id"], group_id=g2["id"])
    assert m2_assigned["group_id"] == g2["id"]
    assert m2_assigned["group"]["name"] == "TEST_MG_Construction"

    # 12. Move Model from Group A to Group B
    m1_moved = update_model(db, m1["id"], group_id=g2["id"])
    assert m1_moved["group_id"] == g2["id"]
    assert m1_moved["group"]["name"] == "TEST_MG_Construction"

    # 13. Ungroup Model
    m1_ungrouped = update_model(db, m1["id"], group_id=None)
    assert m1_ungrouped["group_id"] is None
    assert m1_ungrouped["group"] is None

    # 14, 15, 16. Delete Group (Leaves Models Intact, Sets group_id=null)
    m2_reassigned = update_model(db, m2["id"], group_id=g2["id"])
    assert m2_reassigned["group_id"] == g2["id"]

    del_res = delete_model_group(db, g2["id"])
    assert "deleted successfully" in del_res["message"]

    m2_after_del = get_model(db, m2["id"])
    assert m2_after_del["status"] != "deleted"
    assert m2_after_del["group_id"] is None
    assert m2_after_del["group"] is None

    # 17. Deleted Models Excluded from Group Counts
    g3 = create_model_group(db, name="TEST_MG_CountGroup")
    m3 = create_model(db, name="TEST_MG_Model3", group_id=g3["id"])
    m4 = create_model(db, name="TEST_MG_Model4", group_id=g3["id"])

    g3_info = get_model_group(db, g3["id"])
    assert g3_info["model_count"] == 2

    # Soft-delete m3
    delete_model(db, m3["id"])
    g3_info_after = get_model_group(db, g3["id"])
    assert g3_info_after["model_count"] == 1

    # 18. Historical Inspection Unchanged
    insp_id = "TEST_MG_INSP_999"
    db.ad_inspections.insert_one({
        "id": insp_id,
        "model_id": m4["id"],
        "status": "REJECT",
        "score": 1.85,
        "category": "Screw"
    })
    delete_model_group(db, g3["id"])
    insp_doc = db.ad_inspections.find_one({"id": insp_id})
    assert insp_doc is not None
    assert insp_doc["status"] == "REJECT"
    db.ad_inspections.delete_one({"id": insp_id})

    # 19. Deleted Model Cannot Be Assigned
    res_del_assign = client.patch(f"/api/models/{m3['id']}", json={"group_id": g1['id']})
    assert res_del_assign.status_code == 404

    # 20. Legacy Models without group_id behave as Ungrouped
    legacy_res = db.ad_models.insert_one({"name": "TEST_MG_Legacy", "status": "draft"})
    legacy_model = get_model(db, str(legacy_res.inserted_id))
    assert legacy_model["group_id"] is None
    assert legacy_model["group"] is None
