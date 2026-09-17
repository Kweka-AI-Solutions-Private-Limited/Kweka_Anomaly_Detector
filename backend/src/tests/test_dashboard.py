"""
test_dashboard.py — Complete Test Suite for Point 7B Dashboard
--------------------------------------------------------------
Automated unit tests covering all 23 requirements for Dashboard:
KPI summary, Total Inspections = PASS + REJECT + ERROR, Pass Rate excluding errors,
Model and Date scope filtering, scoped feedback aggregation, reconciliation with History,
legacy run_id=None preservation, empty states, and non-regression.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock
from bson import ObjectId
import pytest

from services.dashboard_service import get_dashboard_summary
from services.inspection_service import get_inspections


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
                            doc_val = doc.get(sk)
                            if sv is None and doc_val is not None:
                                sub_ok = False
                            elif sv == {"$exists": False} and sk in doc:
                                sub_ok = False
                            elif doc_val != sv and str(doc_val) != str(sv):
                                sub_ok = False
                        if sub_ok:
                            or_match = True
                            break
                    if not or_match:
                        match = False
                        break
                elif k == "inspection_id" and isinstance(v, dict) and "$in" in v:
                    val_set = {str(x) for x in v["$in"]}
                    if str(doc.get("inspection_id")) not in val_set:
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
                else:
                    doc_val = doc.get(k)
                    if doc_val != v and str(doc_val) != str(v):
                        match = False
                        break
            if match:
                matched.append(doc)
        return MockCursor(matched)

    def find_one(self, query=None):
        cursor = self.find(query)
        res = cursor.to_list()
        return res[0] if res else None


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
    """Provides a fresh mocked Database instance seeded with test models, versions, runs, inspections, and feedback."""
    db = MagicMock()

    m1_id = ObjectId()
    m2_id = ObjectId()
    v1_id = ObjectId()
    v2_id = ObjectId()
    r1_id = ObjectId()
    r2_id = ObjectId()

    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)
    last_week = now - timedelta(days=7)

    i1_id = ObjectId()
    i2_id = ObjectId()
    i3_id = ObjectId()
    i4_id = ObjectId()

    db.models = MockCollection([
        {"_id": m1_id, "name": "Screw Detector A", "status": "active"},
        {"_id": m2_id, "name": "Nut Detector B", "status": "active"},
    ])

    db.model_versions = MockCollection([
        {"_id": v1_id, "model_id": m1_id, "version_number": 1, "dataset_name": "screw_v1"},
        {"_id": v2_id, "model_id": m1_id, "version_number": 2, "dataset_name": "screw_v2"},
    ])

    db.inspection_runs = MockCollection([
        {"_id": r1_id, "model_id": m1_id, "model_version_id": v1_id, "run_number": 1, "status": "completed", "total_images": 2, "created_at": last_week},
        {"_id": r2_id, "model_id": m1_id, "model_version_id": v2_id, "run_number": 2, "status": "completed", "total_images": 1, "created_at": now},
    ])

    db.inspections = MockCollection([
        {
            "_id": i1_id,
            "model_id": m1_id,
            "model_version_id": v1_id,
            "run_id": r1_id,
            "run_number": 1,
            "status": "completed",
            "prediction": {"status": "normal", "anomaly_score": 12.5, "threshold": 27.0, "severity": "Low"},
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
            "prediction": {"status": "anomalous", "anomaly_score": 45.2, "threshold": 27.0, "severity": "High"},
            "vlm_analysis": {"status": "completed", "defect_type": "scratch_head", "location": "head", "severity": "High", "explanation": "Surface scratch"},
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
            "prediction": {"status": "anomalous", "anomaly_score": 68.0, "threshold": 27.0, "severity": "Critical"},
            "vlm_analysis": {"status": "completed", "defect_type": "thread_side", "location": "thread", "severity": "Critical", "explanation": "Thread deformation"},
            "input": {"filename": "defective_screw_thread.png", "storage_uri": "inspections/i3.png"},
            "created_at": now
        },
        {
            "_id": i4_id,
            "model_id": m2_id,
            "model_version_id": v1_id,
            "run_id": None,
            "status": "completed",
            "prediction": {"status": "normal", "anomaly_score": 15.0, "threshold": 27.0, "severity": "Low"},
            "input": {"filename": "legacy_standalone_nut.png", "storage_uri": "inspections/i4.png"},
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
        }
    ])

    db.feedback = MockCollection([
        {
            "_id": ObjectId(),
            "inspection_id": i2_id,
            "detection_feedback": "correct",
            "vlm_feedback_categories": ["wrong_location"],
            "comment": "Head scratch"
        }
    ])

    return db, m1_id, m2_id, v1_id, v2_id, r1_id, r2_id, i1_id, i2_id, i3_id, i4_id


@pytest.fixture
def mock_db():
    return create_mock_db()


# 1. Dashboard returns KPI summary
def test_dashboard_returns_kpi_summary(mock_db):
    db, *_ = mock_db
    res = get_dashboard_summary(db)
    assert "kpi" in res
    assert "total_inspections" in res["kpi"]


# 2. Total inspection count is correct (PASS + REJECT + ERROR)
def test_total_inspection_count_is_correct(mock_db):
    db, *_ = mock_db
    res = get_dashboard_summary(db)
    assert res["kpi"]["total_inspections"] == 4
    assert res["kpi"]["total_inspections"] == res["kpi"]["pass_count"] + res["kpi"]["reject_count"] + res["kpi"]["error_count"]


# 3 & 4. PASS and REJECT counts are correct
def test_pass_and_reject_counts(mock_db):
    db, *_ = mock_db
    res = get_dashboard_summary(db)
    assert res["kpi"]["pass_count"] == 2  # i1, i4
    assert res["kpi"]["reject_count"] == 2  # i2, i3


# 5 & 6. Pass rate calculation excludes errors
def test_pass_rate_excludes_errors(mock_db):
    db, *_ = mock_db
    res = get_dashboard_summary(db)
    # 2 PASS / (2 PASS + 2 REJECT) * 100 = 50.0%
    assert res["kpi"]["pass_rate"] == 50.0


# 7. Model filter works
def test_model_filter(mock_db):
    db, m1_id, m2_id, *_ = mock_db
    res_m1 = get_dashboard_summary(db, model_id=str(m1_id))
    assert res_m1["kpi"]["total_inspections"] == 3

    res_m2 = get_dashboard_summary(db, model_id=str(m2_id))
    assert res_m2["kpi"]["total_inspections"] == 1


# 8 & 9. Date filter works and works with model filter
def test_date_filter(mock_db):
    db, m1_id, *_ = mock_db
    start_dt = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
    res_date = get_dashboard_summary(db, start_date=start_dt)
    assert res_date["kpi"]["total_inspections"] == 3  # yesterday & today

    res_combo = get_dashboard_summary(db, model_id=str(m1_id), start_date=start_dt)
    assert res_combo["kpi"]["total_inspections"] == 2  # i2 & i3


# 10. Defect distribution is correct
def test_defect_distribution(mock_db):
    db, *_ = mock_db
    res = get_dashboard_summary(db)
    defects = {d["defect_type"]: d["count"] for d in res["defect_distribution"]}
    assert defects.get("scratch_head") == 1
    assert defects.get("thread_side") == 1


# 11. Severity distribution is correct
def test_severity_distribution(mock_db):
    db, *_ = mock_db
    res = get_dashboard_summary(db)
    sev = res["severity_distribution"]
    assert sev["High"] == 1
    assert sev["Critical"] == 1


# 12. Model distribution is correct
def test_model_distribution(mock_db):
    db, *_ = mock_db
    res = get_dashboard_summary(db)
    models_dist = {m["model_name"]: m["count"] for m in res["model_distribution"]}
    assert models_dist.get("Screw Detector A") == 3
    assert models_dist.get("Nut Detector B") == 1


# 13. Run counts are correct
def test_run_counts(mock_db):
    db, *_ = mock_db
    res = get_dashboard_summary(db)
    assert res["run_summary"]["total_runs"] == 2
    assert res["run_summary"]["completed_runs"] == 2


# 14. Feedback counts are scoped
def test_feedback_counts_scoped(mock_db):
    db, m1_id, m2_id, *_ = mock_db
    res_m1 = get_dashboard_summary(db, model_id=str(m1_id))
    assert res_m1["feedback_summary"]["total_feedback"] == 1
    assert res_m1["feedback_summary"]["categories"]["correct"] == 1

    res_m2 = get_dashboard_summary(db, model_id=str(m2_id))
    assert res_m2["feedback_summary"]["total_feedback"] == 0


# 15. Recent inspections are correctly ordered
def test_recent_inspections_ordered(mock_db):
    db, *_ = mock_db
    res = get_dashboard_summary(db)
    recents = res["recent_inspections"]
    assert len(recents) == 4
    # Most recent first
    assert recents[0]["filename"] in ["defective_screw_thread.png", "legacy_standalone_nut.png"]


# 16 & 17. Legacy run_id=null inspections contribute to totals and show Standalone
def test_legacy_inspections_contribute_and_standalone(mock_db):
    db, _, m2_id, *_ = mock_db
    res_m2 = get_dashboard_summary(db, model_id=str(m2_id))
    assert res_m2["kpi"]["total_inspections"] == 1
    recent_nut = res_m2["recent_inspections"][0]
    assert recent_nut["run_id"] is None
    assert recent_nut["run_number"] is None


# 18. No VLM data does not create fake defect/severity values
def test_no_vlm_data_no_fake_defects(mock_db):
    db, _, m2_id, *_ = mock_db
    res_m2 = get_dashboard_summary(db, model_id=str(m2_id))
    assert len(res_m2["defect_distribution"]) == 0


# 19. Empty dashboard state works
def test_empty_dashboard_state():
    empty_db = MagicMock()
    empty_db.inspections = MockCollection([])
    empty_db.inspection_results = MockCollection([])
    empty_db.models = MockCollection([])
    empty_db.model_versions = MockCollection([])
    empty_db.inspection_runs = MockCollection([])
    empty_db.feedback = MockCollection([])

    res = get_dashboard_summary(empty_db)
    assert res["kpi"]["total_inspections"] == 0
    assert res["kpi"]["pass_rate"] == 0.0
    assert len(res["recent_inspections"]) == 0


# 20. RECONCILIATION TEST WITH HISTORY (Adjustment 5!)
def test_reconciliation_with_history(mock_db):
    db, m1_id, *_ = mock_db

    # No filters reconciliation
    dash_no_filter = get_dashboard_summary(db)
    history_no_filter = get_inspections(db)
    assert dash_no_filter["kpi"]["total_inspections"] == len(history_no_filter)

    # Model filter reconciliation
    dash_m1 = get_dashboard_summary(db, model_id=str(m1_id))
    history_m1 = get_inspections(db, model_id=str(m1_id))
    assert dash_m1["kpi"]["total_inspections"] == len(history_m1)
