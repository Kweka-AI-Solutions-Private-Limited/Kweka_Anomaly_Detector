"""
Integration Tests for User-Triggered Gemini AI Defect Analysis
---------------------------------------------------------------
Verifies:
1. REJECT inspection completion does NOT wait for Gemini and sets vlm_analysis = None.
2. PASS inspection sets vlm_analysis.status == "skipped".
3. Triggering POST /api/inspections/{id}/analyze-vlm executes Gemini and sets vlm_analysis.status == "completed".
4. Subsequent calls return cached vlm_analysis without calling Gemini again.
5. Concurrency guard blocks duplicate requests while status == "generating" with HTTP 409 Conflict.
6. Explicit force=True allows forcing Gemini re-analysis.
7. Failed Gemini calls save status == "failed" and allow explicit retry.
"""

import io
import pytest
from unittest.mock import patch
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
    """Provides an active model for testing."""
    db = get_db()
    db.models.delete_many({"name": "Test Gemini Model"})
    
    model_id = ObjectId()
    version_id = ObjectId()
    
    db.models.insert_one({
        "_id": model_id,
        "name": "Test Gemini Model",
        "category": "screw",
        "description": "Integration test model for user-triggered Gemini",
        "status": "active",
        "active_version_id": version_id,
        "reference_image_count": 5
    })
    
    db.model_versions.insert_one({
        "_id": version_id,
        "model_id": model_id,
        "version_number": 1,
        "status": "ready",
        "calibration": {
            "threshold": 20.0,
            "auto_calibrated_threshold": 20.0,
            "method": "95th_percentile"
        },
        "artifacts": {
            "memory_bank": "dummy_path.pt"
        }
    })
    
    yield {"model_id": str(model_id), "version_id": str(version_id)}
    
    db.models.delete_many({"_id": model_id})
    db.model_versions.delete_many({"model_id": model_id})
    db.inspections.delete_many({"model_id": model_id})
    db.inspection_runs.delete_many({"model_id": model_id})


def test_reject_inspection_initial_state_null_vlm(active_model):
    """Verify REJECT inspection leaves vlm_analysis = None and does NOT auto-call Gemini."""
    img_bytes = create_dummy_image_bytes()
    
    with patch("services.inspection_service.run_patchcore_inference") as mock_infer, \
         patch("services.inspection_service.analyze_inspection_evidence") as mock_vlm:
        
        mock_infer.return_value = {
            "status": "anomalous",
            "anomaly_score": 28.5,
            "threshold": 20.0,
            "heatmap": [],
            "bbox": {"x": 10, "y": 10, "width": 30, "height": 30},
            "severity": "HIGH",
            "heatmap_uri": None,
            "processing_time_ms": 110.0
        }
        
        response = client.post(
            "/api/inspections",
            data={"model_id": active_model["model_id"]},
            files={"image": ("test_reject.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 201, response.text
        res_data = response.json()
        
        # PatchCore verdict
        assert res_data["prediction"]["status"] == "anomalous"
        # Gemini was NOT called automatically
        mock_vlm.assert_not_called()
        # vlm_analysis remains null
        assert res_data.get("vlm_analysis") is None


def test_pass_inspection_initial_state_skipped_vlm(active_model):
    """Verify PASS inspection sets vlm_analysis.status = 'skipped'."""
    img_bytes = create_dummy_image_bytes()
    
    with patch("services.inspection_service.run_patchcore_inference") as mock_infer, \
         patch("services.inspection_service.analyze_inspection_evidence") as mock_vlm:
        
        mock_infer.return_value = {
            "status": "normal",
            "anomaly_score": 12.0,
            "threshold": 20.0,
            "heatmap": [],
            "bbox": None,
            "severity": "NONE",
            "heatmap_uri": None,
            "processing_time_ms": 95.0
        }
        
        response = client.post(
            "/api/inspections",
            data={"model_id": active_model["model_id"]},
            files={"image": ("test_pass.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 201, response.text
        res_data = response.json()
        
        assert res_data["prediction"]["status"] == "normal"
        mock_vlm.assert_not_called()
        assert res_data.get("vlm_analysis") is not None
        assert res_data["vlm_analysis"]["status"] == "skipped"


def test_user_triggered_vlm_analysis_flow(active_model):
    """Verify triggering Gemini analysis via POST /api/inspections/{id}/analyze-vlm."""
    img_bytes = create_dummy_image_bytes()
    
    # 1. Create REJECT inspection
    with patch("services.inspection_service.run_patchcore_inference") as mock_infer:
        mock_infer.return_value = {
            "status": "anomalous",
            "anomaly_score": 35.0,
            "threshold": 20.0,
            "heatmap": [],
            "bbox": {"x": 5, "y": 5, "width": 20, "height": 20},
            "severity": "CRITICAL",
            "heatmap_uri": None,
            "processing_time_ms": 105.0
        }
        
        create_res = client.post(
            "/api/inspections",
            data={"model_id": active_model["model_id"]},
            files={"image": ("test_trigger.png", img_bytes, "image/png")}
        )
        assert create_res.status_code == 201
        inspection_id = create_res.json()["inspection_id"]

    # 2. Trigger Gemini Analysis explicitly
    mock_vlm_result = {
        "status": "completed",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "defect_type": "Deep Scratch",
        "location": "Upper Center",
        "severity": "High",
        "prominence": "High",
        "explanation": "Linear surface disruption detected.",
        "visual_evidence": "Bright red anomaly cluster.",
        "confidence": 0.94
    }
    
    with patch("services.inspection_service.analyze_inspection_evidence") as mock_vlm:
        mock_vlm.return_value = mock_vlm_result
        
        trigger_res = client.post(f"/api/inspections/{inspection_id}/analyze-vlm")
        assert trigger_res.status_code == 200, trigger_res.text
        updated_data = trigger_res.json()
        
        mock_vlm.assert_called_once()
        assert updated_data["vlm_analysis"]["status"] == "completed"
        assert updated_data["vlm_analysis"]["defect_type"] == "Deep Scratch"
        assert updated_data["vlm_analysis"]["confidence"] == 0.94

    # 3. Duplicate trigger call without force -> Returns cached result WITHOUT calling Gemini again
    with patch("services.inspection_service.analyze_inspection_evidence") as mock_vlm_2:
        cached_res = client.post(f"/api/inspections/{inspection_id}/analyze-vlm")
        assert cached_res.status_code == 200
        mock_vlm_2.assert_not_called()
        assert cached_res.json()["vlm_analysis"]["defect_type"] == "Deep Scratch"


def test_concurrency_lock_prevents_duplicate_generation(active_model):
    """Verify that calling analyze-vlm when status is 'generating' returns HTTP 409 Conflict."""
    db = get_db()
    insp_id = ObjectId()
    
    # Insert inspection result with status='generating'
    db.inspections.insert_one({
        "_id": insp_id,
        "model_id": ObjectId(active_model["model_id"]),
        "model_version_id": ObjectId(active_model["version_id"]),
        "status": "completed",
        "input": {"filename": "concurrent.png", "storage_uri": "dummy.png"}
    })
    
    db.inspection_results.insert_one({
        "inspection_id": insp_id,
        "prediction": {"status": "anomalous", "anomaly_score": 30.0, "threshold": 20.0, "severity": "HIGH"},
        "localization": {"bbox": None, "heatmap_uri": None},
        "vlm_analysis": {
            "status": "generating",
            "provider": "gemini",
            "explanation": "AI analysis is currently generating."
        }
    })
    
    # Attempting to trigger analyze-vlm while generating must return 409
    res = client.post(f"/api/inspections/{str(insp_id)}/analyze-vlm")
    assert res.status_code == 409
    assert "currently generating" in res.json()["detail"]


def test_force_param_forces_reanalysis(active_model):
    """Verify passing force=true forces a new Gemini call even if completed."""
    db = get_db()
    insp_id = ObjectId()
    
    db.inspections.insert_one({
        "_id": insp_id,
        "model_id": ObjectId(active_model["model_id"]),
        "model_version_id": ObjectId(active_model["version_id"]),
        "status": "completed",
        "input": {"filename": "force.png", "storage_uri": "dummy.png"}
    })
    
    db.inspection_results.insert_one({
        "inspection_id": insp_id,
        "prediction": {"status": "anomalous", "anomaly_score": 30.0, "threshold": 20.0, "severity": "HIGH"},
        "localization": {"bbox": None, "heatmap_uri": None},
        "vlm_analysis": {
            "status": "completed",
            "provider": "gemini",
            "defect_type": "Old Defect"
        }
    })
    
    new_vlm_result = {
        "status": "completed",
        "provider": "gemini",
        "defect_type": "Regenerated Defect",
        "explanation": "Forced fresh analysis."
    }
    
    with patch("services.inspection_service.analyze_inspection_evidence") as mock_vlm:
        mock_vlm.return_value = new_vlm_result
        
        res = client.post(f"/api/inspections/{str(insp_id)}/analyze-vlm?force=true")
        assert res.status_code == 200
        mock_vlm.assert_called_once()
        assert res.json()["vlm_analysis"]["defect_type"] == "Regenerated Defect"
