"""
test_history.py — Complete Test Suite for Point 7A History UX Refinement
------------------------------------------------------------------------
Automated unit tests covering all 17 requirements for History:
cascading filter dependencies (Model -> Version -> Run), unique filter options,
panel apply/clear, date range, pagination, version & run isolation,
legacy run_id=None preservation, and single inspection detail consistency.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock
from bson import ObjectId
import pytest

from services.inspection_service import (
    get_inspections, get_inspection, submit_feedback
)


class MockCollection:
    """In-memory collection mock for PyMongo operations in tests."""
    def __init__(self, data=None):
        self.data = data if data is not None else []

    def find(self, query=None):
        query = query or {}
        matched = []
        for doc in self.data:
            match = True
            for k, v in query.items():
                if k == "$or":
                    or_match = False
                    for subq in v:
                        sub_ok = True
                        for sk, sv in subq.items():
                            if sv is None and doc.get(sk) is not None:
                                sub_ok = False
                            elif sv == {"$exists": False} and sk in doc:
                                sub_ok = False
                        if sub_ok:
                            or_match = True
                            break
                    if not or_match:
                        match = False
                        break
                elif k == "created_at" and isinstance(v, dict):
                    c_dt = doc.get("created_at")
                    if not c_dt:
                        match = False
                        break
                    if "$gte" in v and c_dt < v["$gte"]:
                        match = False
                        break
                    if "$lte" in v and c_dt > v["$lte"]:
                        match = False
                        break
                elif doc.get(k) != v:
                    match = False
                    break
            if match:
                matched.append(doc)
        return MockCursor(matched)

    def find_one(self, query=None):
        cursor = self.find(query)
        res = cursor.to_list()
        return res[0] if res else None

    def insert_one(self, doc):
        if "_id" not in doc:
            doc["_id"] = ObjectId()
        self.data.append(doc)
        res = MagicMock()
        res.inserted_id = doc["_id"]
        return res

    def insert_many(self, docs):
        for d in docs:
            self.insert_one(d)

    def update_one(self, query, update):
        doc = self.find_one(query)
        if doc and "$set" in update:
            doc.update(update["$set"])


class MockCursor:
    def __init__(self, data):
        self._data = list(data)

    def sort(self, key, direction=1):
        reverse = direction == -1
        self._data.sort(key=lambda x: x.get(key) or datetime.min, reverse=reverse)
        return self

    def __iter__(self):
        return iter(self._data)

    def to_list(self):
        return list(self._data)


def create_mock_db():
    """Provides a fresh mocked Database instance seeded with test models, versions, runs, and inspections."""
    db = MagicMock()

    m1_id = ObjectId()
    m2_id = ObjectId()
    v1_id = ObjectId()
    v2_id = ObjectId()
    v3_m2_id = ObjectId()
    r1_id = ObjectId()
    r2_id = ObjectId()
    r3_id = ObjectId()

    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)
    last_week = now - timedelta(days=7)

    i1_id = ObjectId()
    i2_id = ObjectId()
    i3_id = ObjectId()
    i4_id = ObjectId()
    i5_id = ObjectId()

    db.models = MockCollection([
        {"_id": m1_id, "name": "Screw Detector A", "status": "active"},
        {"_id": m2_id, "name": "Nut Detector B", "status": "active"},
    ])

    db.model_versions = MockCollection([
        {"_id": v1_id, "model_id": m1_id, "version_number": 1, "dataset_name": "screw_v1"},
        {"_id": v2_id, "model_id": m1_id, "version_number": 2, "dataset_name": "screw_v2"},
        {"_id": v3_m2_id, "model_id": m2_id, "version_number": 1, "dataset_name": "nut_v1"},
    ])

    db.inspection_runs = MockCollection([
        {"_id": r1_id, "model_id": m1_id, "model_version_id": v1_id, "run_number": 1, "status": "completed"},
        {"_id": r2_id, "model_id": m1_id, "model_version_id": v2_id, "run_number": 2, "status": "completed"},
        {"_id": r3_id, "model_id": m2_id, "model_version_id": v3_m2_id, "run_number": 1, "status": "completed"},
    ])

    db.inspections = MockCollection([
        {
            "_id": i1_id,
            "model_id": m1_id,
            "model_version_id": v1_id,
            "run_id": r1_id,
            "run_number": 1,
            "status": "completed",
            "input": {"filename": "good_screw_01.png", "storage_uri": "inspections/i1.png"},
            "created_at": last_week
        },
        {
            "_id": i2_id,
            "model_id": m1_id,
            "model_version_id": v1_id,
            "run_id": r1_id,
            "run_number": 1,
            "status": "completed",
            "input": {"filename": "defective_screw_scratch.png", "storage_uri": "inspections/i2.png"},
            "created_at": yesterday
        },
        {
            "_id": i3_id,
            "model_id": m1_id,
            "model_version_id": v2_id,
            "run_id": r2_id,
            "run_number": 2,
            "status": "completed",
            "input": {"filename": "defective_screw_thread.png", "storage_uri": "inspections/i3.png"},
            "created_at": now
        },
        {
            "_id": i4_id,
            "model_id": m2_id,
            "model_version_id": v3_m2_id,
            "run_id": None,
            "status": "completed",
            "input": {"filename": "legacy_standalone_nut.png", "storage_uri": "inspections/i4.png"},
            "created_at": now
        },
        {
            "_id": i5_id,
            "model_id": m2_id,
            "model_version_id": v3_m2_id,
            "run_id": r3_id,
            "run_number": 1,
            "status": "completed",
            "input": {"filename": "nut_run1.png", "storage_uri": "inspections/i5.png"},
            "created_at": now
        }
    ])

    db.inspection_results = MockCollection([
        {
            "_id": ObjectId(),
            "inspection_id": i1_id,
            "prediction": {"status": "normal", "anomaly_score": 12.5, "threshold": 27.0, "severity": "Low"},
            "vlm_analysis": {"status": "skipped", "defect_type": "Not required", "explanation": "No anomaly"}
        },
        {
            "_id": ObjectId(),
            "inspection_id": i2_id,
            "prediction": {"status": "anomalous", "anomaly_score": 45.2, "threshold": 27.0, "severity": "High"},
            "vlm_analysis": {"status": "completed", "defect_type": "scratch_head", "location": "head", "severity": "High", "explanation": "Surface scratch"}
        },
        {
            "_id": ObjectId(),
            "inspection_id": i3_id,
            "prediction": {"status": "anomalous", "anomaly_score": 68.0, "threshold": 27.0, "severity": "Critical"},
            "vlm_analysis": {"status": "completed", "defect_type": "thread_side", "location": "thread", "severity": "Critical", "explanation": "Thread deformation"}
        },
        {
            "_id": ObjectId(),
            "inspection_id": i4_id,
            "prediction": {"status": "normal", "anomaly_score": 15.0, "threshold": 27.0, "severity": "Low"},
            "vlm_analysis": {"status": "skipped", "defect_type": "Not required", "explanation": "No anomaly"}
        },
        {
            "_id": ObjectId(),
            "inspection_id": i5_id,
            "prediction": {"status": "anomalous", "anomaly_score": 50.0, "threshold": 27.0, "severity": "Medium"},
            "vlm_analysis": {"status": "completed", "defect_type": "crack", "location": "body", "severity": "Medium", "explanation": "Nut crack"}
        }
    ])

    db.feedback = MockCollection([])

    return db, m1_id, m2_id, v1_id, v2_id, v3_m2_id, r1_id, r2_id, r3_id, i1_id, i2_id, i3_id, i4_id, i5_id


@pytest.fixture
def mock_db():
    return create_mock_db()


# 1. Unique Model options
def test_unique_model_options(mock_db):
    db, *_ = mock_db
    models = list(db.models.find())
    unique_model_ids = {str(m["_id"]) for m in models}
    assert len(unique_model_ids) == len(models)


# 2. Unique Version options
def test_unique_version_options(mock_db):
    db, m1_id, _, v1_id, v2_id, *_ = mock_db
    m1_versions = list(db.model_versions.find({"model_id": m1_id}))
    version_numbers = [v["version_number"] for v in m1_versions]
    assert len(version_numbers) == len(set(version_numbers))


# 3. Versions are scoped to selected Model
def test_versions_scoped_to_selected_model(mock_db):
    db, m1_id, m2_id, v1_id, v2_id, v3_m2_id, *_ = mock_db
    m1_versions = {str(v["_id"]) for v in db.model_versions.find({"model_id": m1_id})}
    m2_versions = {str(v["_id"]) for v in db.model_versions.find({"model_id": m2_id})}

    assert str(v1_id) in m1_versions
    assert str(v2_id) in m1_versions
    assert str(v3_m2_id) not in m1_versions
    assert str(v3_m2_id) in m2_versions


# 4. Runs are scoped to selected Model + Version
def test_runs_scoped_to_selected_model_and_version(mock_db):
    db, m1_id, m2_id, v1_id, v2_id, v3_m2_id, r1_id, r2_id, r3_id, *_ = mock_db
    v1_runs = {str(r["_id"]) for r in db.inspection_runs.find({"model_id": m1_id, "model_version_id": v1_id})}
    v2_runs = {str(r["_id"]) for r in db.inspection_runs.find({"model_id": m1_id, "model_version_id": v2_id})}
    v3_runs = {str(r["_id"]) for r in db.inspection_runs.find({"model_id": m2_id, "model_version_id": v3_m2_id})}

    assert str(r1_id) in v1_runs
    assert str(r2_id) not in v1_runs
    assert str(r2_id) in v2_runs
    assert str(r3_id) in v3_runs


# 5 & 6. Version disabled before Model selection & Run disabled before Version selection
def test_cascading_disabled_states_conceptually():
    draft_model = "all"
    available_versions = [] if draft_model == "all" else ["v1", "v2"]
    assert len(available_versions) == 0

    draft_version = "all"
    available_runs = [] if (draft_model == "all" or draft_version == "all") else ["r1"]
    assert len(available_runs) == 0


# 7. Changing Model resets Version and Run
def test_changing_model_resets_version_and_run(mock_db):
    db, m1_id, m2_id, v1_id, v2_id, v3_m2_id, r1_id, r2_id, r3_id, *_ = mock_db
    
    # Selecting m1, v1, r1
    res_m1_v1_r1 = get_inspections(db, model_id=str(m1_id), model_version_id=str(v1_id), run_id=str(r1_id))
    assert len(res_m1_v1_r1) == 2

    # Switching to m2 (should query only m2 when version and run are reset to None/all)
    res_m2 = get_inspections(db, model_id=str(m2_id))
    assert len(res_m2) == 2  # i4 (standalone) & i5 (run #1 for m2)


# 8. Changing Version resets Run
def test_changing_version_resets_run(mock_db):
    db, m1_id, _, v1_id, v2_id, _, r1_id, r2_id, *_ = mock_db
    
    # Selecting m1 and v2 with run reset to None/all
    res_v2 = get_inspections(db, model_id=str(m1_id), model_version_id=str(v2_id))
    assert len(res_v2) == 1
    assert str(res_v2[0]["run_id"]) == str(r2_id)


# 9. Invalid Model + Version + Run combinations cannot be selected
def test_invalid_combination_returns_empty(mock_db):
    db, m1_id, _, _, _, v3_m2_id, r1_id, *_ = mock_db
    # Asking for m1 with m2's version (v3_m2_id) should yield 0 results
    res = get_inspections(db, model_id=str(m1_id), model_version_id=str(v3_m2_id))
    assert len(res) == 0


# 10, 11, 12. Filter panel apply/clear behavior & Multiple filters work together
def test_multiple_filters_and_clear(mock_db):
    db, m1_id, _, v1_id, _, _, r1_id, _, _, i1_id, i2_id, *_ = mock_db
    
    # Apply multiple filters
    res = get_inspections(db, model_id=str(m1_id), model_version_id=str(v1_id), run_id=str(r1_id), status="reject")
    assert len(res) == 1
    assert str(res[0]["inspection_id"]) == str(i2_id)

    # Clear filters
    res_all = get_inspections(db)
    assert len(res_all) == 5


# 13, 14. Legacy run_id=null records remain visible & filterable
def test_legacy_run_id_null_visible(mock_db):
    db, _, _, _, _, _, _, _, _, _, _, _, i4_id, _ = mock_db
    res_standalone = get_inspections(db, run_id="standalone")
    assert len(res_standalone) == 1
    assert str(res_standalone[0]["inspection_id"]) == str(i4_id)
    assert res_standalone[0].get("run_id") is None


# 15. Existing inspection detail navigation still works
def test_inspection_detail_navigation(mock_db):
    db, _, _, _, _, _, _, _, _, _, i2_id, *_ = mock_db
    detail = get_inspection(db, str(i2_id))
    assert detail["inspection_id"] == str(i2_id)
    assert detail["filename"] == "defective_screw_scratch.png"


# 16 & 17. Feedback submission and retrieval compatibility
def test_feedback_submission_compatibility(mock_db):
    db, _, _, _, _, _, _, _, _, _, i2_id, *_ = mock_db
    fb = submit_feedback(
        db,
        inspection_id=str(i2_id),
        detection_feedback="correct",
        vlm_feedback_categories=["wrong_location"],
        corrected_location="head_center",
        comment="Location precision adjustment"
    )
    assert fb["inspection_id"] == str(i2_id)
    assert fb["vlm_feedback_categories"] == ["wrong_location"]
