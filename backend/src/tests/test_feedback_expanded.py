import sys
from pathlib import Path

# Add backend/src to sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from api_server import app
from db.connection import get_db
db = get_db()
from db.schemas import (
    InspectionSchema,
    PredictionOutput,
    VLMAnalysisSchema,
    ModelSchema,
    ModelVersionSchema,
    InspectionRunSchema,
)

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_db():
    """Setup clean test data in MongoDB before test execution."""
    # Clean up test collections
    db["models"].delete_many({"name": "Test Feedback Model"})
    db["model_versions"].delete_many({"version_number": 99})
    db["inspection_runs"].delete_many({"run_number": 999})
    db["inspections"].delete_many({"filename": "test_feedback_sample.png"})
    db["feedback"].delete_many({"comment": {"$regex": "Test Feedback"}})

    # Create parent hierarchy
    model_res = db["models"].insert_one({
        "name": "Test Feedback Model",
        "description": "Model for feedback test suite",
        "reference_image_count": 5,
        "created_at": datetime.utcnow(),
    })
    model_id = str(model_res.inserted_id)

    version_res = db["model_versions"].insert_one({
        "model_id": model_id,
        "version_number": 99,
        "status": "active",
        "parameters": {},
        "created_at": datetime.utcnow(),
    })
    version_id = str(version_res.inserted_id)

    run_res = db["inspection_runs"].insert_one({
        "model_id": model_id,
        "model_version_id": version_id,
        "run_number": 999,
        "status": "completed",
        "total_images": 2,
        "completed_images": 2,
        "flagged_anomalies": 1,
        "created_at": datetime.utcnow(),
    })
    run_id = str(run_res.inserted_id)

    # Create REJECT inspection sample
    reject_insp_res = db["inspections"].insert_one({
        "model_id": model_id,
        "model_version_id": version_id,
        "run_id": run_id,
        "filename": "test_feedback_sample.png",
        "storage_uri": "uploads/test_reject.png",
        "prediction": {
            "is_anomalous": True,
            "anomaly_score": 35.5,
            "threshold": 25.0,
            "status": "reject",
            "vlm_analysis": {
                "defect_type": "Scratch",
                "location": "Top Left",
                "severity": "High",
                "prominence": "Very prominent surface line",
                "explanation": "Clear surface defect detected"
            }
        },
        "created_at": datetime.utcnow(),
    })
    reject_insp_id = str(reject_insp_res.inserted_id)

    # Create PASS inspection sample
    pass_insp_res = db["inspections"].insert_one({
        "model_id": model_id,
        "model_version_id": version_id,
        "run_id": run_id,
        "filename": "test_feedback_sample.png",
        "storage_uri": "uploads/test_pass.png",
        "prediction": {
            "is_anomalous": False,
            "anomaly_score": 12.0,
            "threshold": 25.0,
            "status": "normal"
        },
        "created_at": datetime.utcnow(),
    })
    pass_insp_id = str(pass_insp_res.inserted_id)

    yield {
        "model_id": model_id,
        "version_id": version_id,
        "run_id": run_id,
        "reject_insp_id": reject_insp_id,
        "pass_insp_id": pass_insp_id,
    }

    # Clean up after tests
    db["models"].delete_many({"_id": model_res.inserted_id})
    db["model_versions"].delete_many({"_id": version_res.inserted_id})
    db["inspection_runs"].delete_many({"_id": run_res.inserted_id})
    db["inspections"].delete_many({"filename": "test_feedback_sample.png"})
    db["feedback"].delete_many({"comment": {"$regex": "Test Feedback"}})


def test_submit_valid_reject_feedback(setup_test_db):
    """Test submitting valid feedback for REJECT inspection result."""
    data = setup_test_db
    payload = {
        "detection_feedback": "correct",
        "vlm_feedback_categories": ["correct"],
        "comment": "Test Feedback: REJECT is correct"
    }

    response = client.post(f"/api/inspections/{data['reject_insp_id']}/feedback", json=payload)
    assert response.status_code in (200, 201)
    res_data = response.json()
    assert res_data["detection_feedback"] == "correct"
    assert res_data["vlm_feedback_categories"] == ["correct"]
    assert res_data["model_id"] == data["model_id"]
    assert res_data["model_version_id"] == data["version_id"]
    assert res_data["run_id"] == data["run_id"]
    assert res_data["original_prediction"]["status"] == "reject"


def test_submit_reject_false_positive(setup_test_db):
    """Test submitting False Positive feedback for REJECT result."""
    data = setup_test_db
    payload = {
        "detection_feedback": "false_positive",
        "vlm_feedback_categories": ["wrong_defect_type"],
        "corrected_defect_type": "Dust Particle",
        "comment": "Test Feedback: Image is actually clean, just dust"
    }

    response = client.post(f"/api/inspections/{data['reject_insp_id']}/feedback", json=payload)
    assert response.status_code in (200, 201)
    res_data = response.json()
    assert res_data["detection_feedback"] == "false_positive"
    assert res_data["corrected_defect_type"] == "Dust Particle"


def test_reject_invalid_detection_feedback_on_reject(setup_test_db):
    """Verify that False Negative is rejected on a REJECT inspection."""
    data = setup_test_db
    payload = {
        "detection_feedback": "false_negative",  # Invalid for REJECT!
        "comment": "Test Feedback Invalid"
    }

    response = client.post(f"/api/inspections/{data['reject_insp_id']}/feedback", json=payload)
    assert response.status_code == 400
    assert "cannot be 'false_negative'" in response.json()["detail"]


def test_submit_valid_pass_feedback(setup_test_db):
    """Test submitting valid feedback for PASS inspection result."""
    data = setup_test_db
    payload = {
        "detection_feedback": "correct",
        "vlm_feedback_categories": ["correct"],
        "comment": "Test Feedback: PASS is correct"
    }

    response = client.post(f"/api/inspections/{data['pass_insp_id']}/feedback", json=payload)
    assert response.status_code in (200, 201)
    res_data = response.json()
    assert res_data["detection_feedback"] == "correct"


def test_submit_pass_false_negative(setup_test_db):
    """Test submitting False Negative feedback for PASS result."""
    data = setup_test_db
    payload = {
        "detection_feedback": "false_negative",
        "corrected_defect_type": "Micro Crack",
        "comment": "Test Feedback: Missed subtle micro crack on component"
    }

    response = client.post(f"/api/inspections/{data['pass_insp_id']}/feedback", json=payload)
    assert response.status_code in (200, 201)
    res_data = response.json()
    assert res_data["detection_feedback"] == "false_negative"
    assert res_data["corrected_defect_type"] == "Micro Crack"


def test_reject_invalid_detection_feedback_on_pass(setup_test_db):
    """Verify that False Positive is rejected on a PASS inspection."""
    data = setup_test_db
    payload = {
        "detection_feedback": "false_positive",  # Invalid for PASS!
        "comment": "Test Feedback Invalid"
    }

    response = client.post(f"/api/inspections/{data['pass_insp_id']}/feedback", json=payload)
    assert response.status_code == 400
    assert "cannot be 'false_positive'" in response.json()["detail"]


def test_non_destructive_prediction_preservation(setup_test_db):
    """Verify submitting feedback does NOT alter original inspection record in DB."""
    data = setup_test_db
    payload = {
        "detection_feedback": "false_positive",
        "comment": "Test Feedback Non Destructive"
    }

    client.post(f"/api/inspections/{data['reject_insp_id']}/feedback", json=payload)

    # Fetch original inspection from API
    response = client.get(f"/api/inspections/{data['reject_insp_id']}")
    assert response.status_code == 200
    insp = response.json()

    # Original prediction status MUST STILL BE REJECT!
    assert insp["prediction"]["status"] == "reject"
    assert insp["prediction"]["is_anomalous"] is True
    assert insp["prediction"]["anomaly_score"] == 35.5


def test_workspace_feedback_filtering(setup_test_db):
    """Verify workspace GET /api/feedback filters data correctly."""
    data = setup_test_db

    # Submit feedback records
    client.post(f"/api/inspections/{data['reject_insp_id']}/feedback", json={
        "detection_feedback": "false_positive",
        "comment": "Test Feedback Workspace FP"
    })
    client.post(f"/api/inspections/{data['pass_insp_id']}/feedback", json={
        "detection_feedback": "correct",
        "comment": "Test Feedback Workspace Pass Correct"
    })

    # 1. Filter by Model ID
    res_model = client.get(f"/api/feedback?model_id={data['model_id']}")
    assert res_model.status_code == 200
    records = res_model.json()
    assert len(records) >= 2

    # 2. Filter by Feedback Type
    res_fp = client.get(f"/api/feedback?feedback_type=false_positive")
    assert res_fp.status_code == 200
    fp_records = res_fp.json()
    assert all(r["detection_feedback"] == "false_positive" or r["feedback_type"] == "false_positive" for r in fp_records)
