"""
test_hotspot_analysis.py — Unit Tests for Anomaly Hotspot Analysis (5x5 Grid BBox Overlap)
-----------------------------------------------------------------------------------------
Tests for 5x5 spatial grid hotspot calculation based on persisted PatchCore localization.bbox data.
Verifies:
- Area overlap calculation across 5x5 grid cells.
- Bbox spanning multiple grid cells increments all overlapping cells.
- Scoping by model_id and date range.
- Empty states and missing bbox handling.
- Backwards compatibility (defect_distribution still present).
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock
from bson import ObjectId
import pytest

from services.dashboard_service import get_dashboard_summary


class MockCollection:
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


def create_hotspot_mock_db():
    db = MagicMock()
    m1_id = ObjectId()
    v1_id = ObjectId()
    now = datetime.utcnow()

    # Bbox 1: x=0, y=0, w=100, h=100 (overlaps row 0, col 0 AND row 0, col 1 AND row 1, col 0 AND row 1, col 1 since cell size is 51.2x51.2)
    # Bbox 2: x=200, y=200, w=50, h=50 (overlaps row 3, col 3 AND row 4, col 3 AND row 3, col 4 AND row 4, col 4)
    i1_id = ObjectId()
    i2_id = ObjectId()
    i3_id = ObjectId()  # PASS (no bbox hotspot addition)
    i4_id = ObjectId()  # REJECT without bbox

    db.models = MockCollection([
        {"_id": m1_id, "name": "Hotspot Model A", "status": "active"},
    ])
    db.model_versions = MockCollection([
        {"_id": v1_id, "model_id": m1_id, "version_number": 1},
    ])
    db.inspection_runs = MockCollection([])
    db.feedback = MockCollection([])

    db.inspections = MockCollection([
        {
            "_id": i1_id,
            "model_id": m1_id,
            "model_version_id": v1_id,
            "status": "completed",
            "prediction": {"status": "anomalous", "anomaly_score": 45.0, "threshold": 27.0},
            "created_at": now
        },
        {
            "_id": i2_id,
            "model_id": m1_id,
            "model_version_id": v1_id,
            "status": "completed",
            "prediction": {"status": "anomalous", "anomaly_score": 50.0, "threshold": 27.0},
            "created_at": now
        },
        {
            "_id": i3_id,
            "model_id": m1_id,
            "model_version_id": v1_id,
            "status": "completed",
            "prediction": {"status": "normal", "anomaly_score": 10.0, "threshold": 27.0},
            "created_at": now
        },
        {
            "_id": i4_id,
            "model_id": m1_id,
            "model_version_id": v1_id,
            "status": "completed",
            "prediction": {"status": "anomalous", "anomaly_score": 40.0, "threshold": 27.0},
            "created_at": now
        }
    ])

    db.inspection_results = MockCollection([
        {
            "_id": ObjectId(),
            "inspection_id": i1_id,
            "prediction": {"status": "anomalous", "anomaly_score": 45.0, "threshold": 27.0},
            "localization": {"bbox": {"x": 0, "y": 0, "width": 100, "height": 100}}
        },
        {
            "_id": ObjectId(),
            "inspection_id": i2_id,
            "prediction": {"status": "anomalous", "anomaly_score": 50.0, "threshold": 27.0},
            "localization": {"bbox": {"x": 200, "y": 200, "width": 50, "height": 50}}
        },
        {
            "_id": ObjectId(),
            "inspection_id": i3_id,
            "prediction": {"status": "normal", "anomaly_score": 10.0, "threshold": 27.0},
            "localization": {"bbox": None}
        },
        {
            "_id": ObjectId(),
            "inspection_id": i4_id,
            "prediction": {"status": "anomalous", "anomaly_score": 40.0, "threshold": 27.0},
            "localization": {"bbox": None}
        }
    ])

    return db, m1_id, i1_id, i2_id, i3_id, i4_id


def test_hotspot_analysis_structure():
    db, m1_id, *_ = create_hotspot_mock_db()
    res = get_dashboard_summary(db)

    assert "anomaly_hotspot_analysis" in res
    hotspot = res["anomaly_hotspot_analysis"]

    assert hotspot["grid_size"] == 5
    assert hotspot["total_anomalous_inspections"] == 3  # i1, i2, i4
    assert hotspot["inspections_with_bbox"] == 2  # i1, i2
    assert len(hotspot["matrix"]) == 5
    assert all(len(row) == 5 for row in hotspot["matrix"])
    assert len(hotspot["cells"]) == 25


def test_bbox_area_overlap_across_multiple_cells():
    db, m1_id, *_ = create_hotspot_mock_db()
    res = get_dashboard_summary(db)
    matrix = res["anomaly_hotspot_analysis"]["matrix"]

    # Cell size is 51.2 x 51.2
    # Bbox 1 (0, 0, 100, 100) spans:
    # x in [0, 100] -> cols 0 (0..51.2) and 1 (51.2..102.4)
    # y in [0, 100] -> rows 0 (0..51.2) and 1 (51.2..102.4)
    assert matrix[0][0] == 1
    assert matrix[0][1] == 1
    assert matrix[1][0] == 1
    assert matrix[1][1] == 1

    # Bbox 2 (200, 200, 50, 50) spans:
    # x in [200, 250] -> col 3 (153.6..204.8) and col 4 (204.8..256.0)
    # y in [200, 250] -> row 3 (153.6..204.8) and row 4 (204.8..256.0)
    assert matrix[3][3] == 1
    assert matrix[3][4] == 1
    assert matrix[4][3] == 1
    assert matrix[4][4] == 1

    # Unaffected cell
    assert matrix[2][2] == 0


def test_defect_distribution_preserved():
    db, *_ = create_hotspot_mock_db()
    res = get_dashboard_summary(db)
    assert "defect_distribution" in res


def test_empty_hotspot_state():
    empty_db = MagicMock()
    empty_db.inspections = MockCollection([])
    empty_db.inspection_results = MockCollection([])
    empty_db.models = MockCollection([])
    empty_db.model_versions = MockCollection([])
    empty_db.inspection_runs = MockCollection([])
    empty_db.feedback = MockCollection([])

    res = get_dashboard_summary(empty_db)
    hotspot = res["anomaly_hotspot_analysis"]
    assert hotspot["total_anomalous_inspections"] == 0
    assert hotspot["inspections_with_bbox"] == 0
    assert hotspot["max_cell_count"] == 0
    assert all(all(val == 0 for val in row) for row in hotspot["matrix"])
