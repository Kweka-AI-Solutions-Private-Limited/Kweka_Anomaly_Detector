"""
Regression Test Suite: Database Routing Override Security
-----------------------------------------------------------
Proves that passing arbitrary db_name query parameters (e.g. ?db_name=anomaly_detection
or ?db_name=malicious_db) DOES NOT switch or redirect the MongoDB database,
and that all endpoints resolve strictly against the configured MONGODB_DATABASE ('Anomaly_Detector').
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure src/ is on Python path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from api_server import app
from db.connection import get_db, MONGODB_DATABASE

client = TestClient(app)


def test_get_db_ignores_arbitrary_db_name_parameter():
    """Direct test verifying get_db() returns client[MONGODB_DATABASE] regardless of input."""
    db1 = get_db()
    db2 = get_db("anomaly_detection")
    db3 = get_db("arbitrary_hacker_db")

    assert db1.name == MONGODB_DATABASE
    assert db2.name == MONGODB_DATABASE
    assert db3.name == MONGODB_DATABASE


def test_inspection_endpoint_ignores_db_name_override():
    """
    Verifies GET /api/inspections/{id} resolves against Anomaly_Detector
    and returns HTTP 200 OK both with and without ?db_name=anomaly_detection.
    """
    db = get_db()
    sample_insp = db.inspections.find_one({})
    if not sample_insp:
        pytest.skip("No inspection records in database to test endpoint.")

    insp_id = str(sample_insp["_id"])

    # 1. Request WITHOUT db_name query parameter
    res1 = client.get(f"/api/inspections/{insp_id}")
    assert res1.status_code == 200, f"Expected 200 without query param, got {res1.status_code}"
    data1 = res1.json()

    # 2. Request WITH ?db_name=anomaly_detection (the original buggy override)
    res2 = client.get(f"/api/inspections/{insp_id}?db_name=anomaly_detection")
    assert res2.status_code == 200, f"Expected 200 with db_name=anomaly_detection, got {res2.status_code}"
    data2 = res2.json()

    # 3. Request WITH ?db_name=random_db
    res3 = client.get(f"/api/inspections/{insp_id}?db_name=random_db")
    assert res3.status_code == 200, f"Expected 200 with db_name=random_db, got {res3.status_code}"
    data3 = res3.json()

    assert data1["inspection_id"] == insp_id
    assert data2["inspection_id"] == insp_id
    assert data3["inspection_id"] == insp_id


def test_feedback_endpoint_ignores_db_name_override():
    """
    Verifies GET /api/inspections/{id}/feedback resolves against Anomaly_Detector
    and returns HTTP 200 OK both with and without ?db_name=anomaly_detection.
    """
    db = get_db()
    sample_insp = db.inspections.find_one({})
    if not sample_insp:
        pytest.skip("No inspection records in database to test endpoint.")

    insp_id = str(sample_insp["_id"])

    # 1. Request WITHOUT db_name query parameter
    res1 = client.get(f"/api/inspections/{insp_id}/feedback")
    assert res1.status_code == 200

    # 2. Request WITH ?db_name=anomaly_detection
    res2 = client.get(f"/api/inspections/{insp_id}/feedback?db_name=anomaly_detection")
    assert res2.status_code == 200

    # 3. Request WITH ?db_name=custom_fake_db
    res3 = client.get(f"/api/inspections/{insp_id}/feedback?db_name=custom_fake_db")
    assert res3.status_code == 200

    assert res1.json() == res2.json() == res3.json()
