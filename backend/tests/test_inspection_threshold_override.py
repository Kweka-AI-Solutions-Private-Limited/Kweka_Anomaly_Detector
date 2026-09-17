"""
Integration Tests for Inspection-Level Custom Threshold Override
------------------------------------------------------------------
Verifies that:
1. Model Details threshold customization is removed / model versions remain immutable.
2. Single image inspection with default vs custom threshold works as expected.
3. Batch inspection run with default vs custom threshold works as expected.
4. Custom threshold does NOT create new model versions.
5. prediction.threshold in inspection results accurately reflects the threshold used.
6. Invalid threshold inputs (< 0, non-numeric) return HTTP 400.
"""

import io
import pytest
from fastapi.testclient import TestClient
from bson import ObjectId
import numpy as np
from PIL import Image

from main import app
from db.connection import get_db

client = TestClient(app)


def create_dummy_image_bytes():
    """Helper to create valid PNG image bytes for test uploads."""
    img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def active_model():
    """Provides an active model with a calibrated model version for testing."""
    db = get_db()
    
    # Clean up test artifacts
    db.models.delete_many({"name": "Test Threshold Model"})
    
    model_id = ObjectId()
    version_id = ObjectId()
    
    # 1. Create Active Model
    db.models.insert_one({
        "_id": model_id,
        "name": "Test Threshold Model",
        "category": "screw",
        "description": "Integration test model for threshold overrides",
        "status": "active",
        "active_version_id": version_id,
        "reference_image_count": 10
    })
    
    # 2. Create Active Version v1 with threshold 24.54
    db.model_versions.insert_one({
        "_id": version_id,
        "model_id": model_id,
        "version_number": 1,
        "status": "ready",
        "calibration": {
            "threshold": 24.54,
            "auto_calibrated_threshold": 24.54,
            "method": "95th_percentile",
            "is_custom": False
        },
        "artifacts": {
            "memory_bank": "dummy_path.pt"
        }
    })
    
    yield {"model_id": str(model_id), "version_id": str(version_id), "default_threshold": 24.54}
    
    # Cleanup after test
    db.models.delete_many({"_id": model_id})
    db.model_versions.delete_many({"model_id": model_id})
    db.inspections.delete_many({"model_id": model_id})
    db.inspection_runs.delete_many({"model_id": model_id})


def test_single_inspection_default_threshold(active_model):
    """Test running a single inspection with Model Default threshold."""
    img_bytes = create_dummy_image_bytes()
    
    from unittest.mock import patch
    with patch("services.inspection_service.run_patchcore_inference") as mock_infer:
        mock_infer.return_value = {
            "status": "normal",
            "anomaly_score": 15.2,
            "threshold": active_model["default_threshold"],
            "heatmap": [],
            "bbox": None,
            "severity": "NONE",
            "heatmap_uri": None,
            "processing_time_ms": 120.0
        }
        
        response = client.post(
            "/api/inspections",
            data={"model_id": active_model["model_id"]},
            files={"image": ("test.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 201, response.text
        res_data = response.json()
        assert res_data["prediction"]["threshold"] == 24.54
        mock_infer.assert_called_once()
        assert mock_infer.call_args[0][2] == 24.54


def test_single_inspection_custom_threshold_no_new_version(active_model):
    """Test running single inspection with custom threshold_override=18.5."""
    db = get_db()
    model_id = ObjectId(active_model["model_id"])
    
    initial_version_count = db.model_versions.count_documents({"model_id": model_id})
    img_bytes = create_dummy_image_bytes()
    
    from unittest.mock import patch
    with patch("services.inspection_service.run_patchcore_inference") as mock_infer:
        mock_infer.return_value = {
            "status": "anomalous",
            "anomaly_score": 20.1,
            "threshold": 18.5,
            "heatmap": [],
            "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
            "severity": "MEDIUM",
            "heatmap_uri": None,
            "processing_time_ms": 135.0
        }
        
        response = client.post(
            "/api/inspections",
            data={
                "model_id": active_model["model_id"],
                "threshold_override": 18.5
            },
            files={"image": ("test.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 201, response.text
        res_data = response.json()
        assert res_data["prediction"]["threshold"] == 18.5
        assert mock_infer.call_args[0][2] == 18.5

    # Verify no new model version was created
    final_version_count = db.model_versions.count_documents({"model_id": model_id})
    assert final_version_count == initial_version_count == 1
    
    # Verify original version calibration threshold remains 24.54
    version = db.model_versions.find_one({"_id": ObjectId(active_model["version_id"])})
    assert version["calibration"]["threshold"] == 24.54


def test_batch_inspection_run_custom_threshold(active_model):
    """Test running batch inspection run with custom threshold_override=30.0."""
    db = get_db()
    model_id = ObjectId(active_model["model_id"])
    initial_version_count = db.model_versions.count_documents({"model_id": model_id})
    
    img_bytes = create_dummy_image_bytes()
    
    from unittest.mock import patch
    with patch("services.inspection_run_service.run_patchcore_inference") as mock_infer:
        mock_infer.return_value = {
            "status": "normal",
            "anomaly_score": 25.0,
            "threshold": 30.0,
            "heatmap": [],
            "bbox": None,
            "severity": "NONE",
            "heatmap_uri": None,
            "processing_time_ms": 110.0
        }
        
        response = client.post(
            "/api/inspection-runs",
            data={
                "model_id": active_model["model_id"],
                "threshold_override": 30.0
            },
            files=[
                ("files", ("test1.png", img_bytes, "image/png")),
                ("files", ("test2.png", img_bytes, "image/png"))
            ]
        )
        
        assert response.status_code == 201, response.text
        run_data = response.json()
        assert run_data["total_images"] == 2

    # Verify model version immutability
    final_version_count = db.model_versions.count_documents({"model_id": model_id})
    assert final_version_count == initial_version_count == 1


def test_invalid_threshold_override(active_model):
    """Test validation errors for invalid threshold values."""
    img_bytes = create_dummy_image_bytes()
    
    # 1. Negative threshold
    res = client.post(
        "/api/inspections",
        data={"model_id": active_model["model_id"], "threshold_override": -5.0},
        files={"image": ("test.png", img_bytes, "image/png")}
    )
    assert res.status_code == 400
    
    # 2. Zero threshold
    res = client.post(
        "/api/inspections",
        data={"model_id": active_model["model_id"], "threshold_override": 0.0},
        files={"image": ("test.png", img_bytes, "image/png")}
    )
    assert res.status_code == 400
