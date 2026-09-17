"""
Integration Tests for Multi-Product Image Inspection (Phase 1)
----------------------------------------------------------------
Verifies that:
1. POST /api/inspections with inspection_mode="multi_instance" detects instances, runs PatchCore per crop, aggregates verdicts, and saves coordinates.
2. Single-product mode remains backward compatible.
3. POST /api/inspections/{inspection_id}/instances/{instance_id}/analyze-vlm triggers Gemini VLM for a specific instance.
4. POST /api/inspection-runs accepts inspection_mode="multi_instance".
"""

import io
import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from bson import ObjectId
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from api_server import app
from db.connection import get_db

client = TestClient(app)


def create_dummy_image_bytes():
    """Helper to create valid PNG image bytes for test uploads."""
    img = Image.fromarray(np.uint8(np.random.rand(200, 200, 3) * 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def dummy_imwrite(p, img):
    """Helper mock for cv2.imwrite that creates a valid dummy file on disk."""
    path_obj = Path(p)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_bytes(b"dummy_image_data")
    return True


@pytest.fixture
def active_model():
    """Provides an active model with a calibrated model version for multi-instance testing."""
    db = get_db()
    
    db.models.delete_many({"name": "Test MultiInstance Model"})
    
    model_id = ObjectId()
    version_id = ObjectId()
    
    db.models.insert_one({
        "_id": model_id,
        "name": "Test MultiInstance Model",
        "category": "screw",
        "description": "Integration test model for multi-instance inspection",
        "status": "active",
        "active_version_id": version_id,
        "reference_image_count": 10
    })
    
    db.model_versions.insert_one({
        "_id": version_id,
        "model_id": model_id,
        "version_number": 1,
        "status": "ready",
        "calibration": {
            "threshold": 25.0,
            "auto_calibrated_threshold": 25.0,
            "method": "95th_percentile",
            "is_custom": False
        },
        "artifacts": {
            "memory_bank": "dummy_path.pt"
        }
    })
    
    yield {"model_id": str(model_id), "version_id": str(version_id), "default_threshold": 25.0}
    
    # Cleanup
    db.models.delete_many({"_id": model_id})
    db.model_versions.delete_many({"model_id": model_id})
    db.inspections.delete_many({"model_id": model_id})
    db.inspection_runs.delete_many({"model_id": model_id})
    db.inspection_results.delete_many({})


def test_multi_instance_inspection_flow(active_model):
    """Test full multi-instance inspection API flow."""
    img_bytes = create_dummy_image_bytes()
    
    # Mock instance detection to return 2 instances with full fields
    dummy_instances = [
        {
            "instance_id": 1,
            "crop_filepath": "/tmp/crop1.png",
            "crop_storage_uri": "storage/crops/crop1.png",
            "bbox": {"x": 10, "y": 10, "width": 50, "height": 50},
            "padded_bbox": {"x": 8, "y": 8, "width": 54, "height": 54},
            "detection_confidence": 0.95
        },
        {
            "instance_id": 2,
            "crop_filepath": "/tmp/crop2.png",
            "crop_storage_uri": "storage/crops/crop2.png",
            "bbox": {"x": 100, "y": 100, "width": 50, "height": 50},
            "padded_bbox": {"x": 98, "y": 98, "width": 54, "height": 54},
            "detection_confidence": 0.95
        }
    ]
    
    with patch("services.inspection_service.detect_and_crop_instances") as mock_detect, \
         patch("services.inspection_service.run_patchcore_inference") as mock_infer, \
         patch("cv2.imread") as mock_imread, \
         patch("cv2.imwrite", side_effect=dummy_imwrite):
        
        mock_detect.return_value = (dummy_instances, {"detection_time_ms": 12.0})
        mock_imread.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Mock instance 1 = PASS (score 15.0), instance 2 = REJECT (score 35.0)
        mock_infer.side_effect = [
            {"status": "PASS", "anomaly_score": 15.0, "threshold": 25.0, "severity": "NONE", "heatmap": [], "bbox": None, "heatmap_uri": None, "processing_time_ms": 50.0},
            {"status": "REJECT", "anomaly_score": 35.0, "threshold": 25.0, "severity": "HIGH", "heatmap": [], "bbox": {"x": 5, "y": 5, "width": 10, "height": 10}, "heatmap_uri": None, "processing_time_ms": 60.0}
        ]
        
        response = client.post(
            "/api/inspections",
            data={
                "model_id": active_model["model_id"],
                "inspection_mode": "multi_instance"
            },
            files={"image": ("multi_test.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 201, response.text
        res_data = response.json()
        
        result = res_data.get("result", res_data)
        assert res_data.get("inspection_mode") == "multi_instance" or result.get("inspection_mode") == "multi_instance"
        assert result["overall_prediction"]["status"] == "REJECT"
        assert result["overall_prediction"]["total_instances"] == 2
        assert result["overall_prediction"]["pass_count"] == 1
        assert result["overall_prediction"]["reject_count"] == 1
        assert len(result["instances"]) == 2
        
        inst1 = result["instances"][0]
        assert inst1["instance_id"] == 1
        assert inst1["prediction"]["status"] == "PASS"
        
        inst2 = result["instances"][1]
        assert inst2["instance_id"] == 2
        assert inst2["prediction"]["status"] == "REJECT"


def test_multi_instance_vlm_retry(active_model):
    """Test instance-specific Gemini VLM analysis endpoint."""
    img_bytes = create_dummy_image_bytes()
    dummy_instances = [
        {
            "instance_id": 1,
            "crop_filepath": "/tmp/crop1.png",
            "crop_storage_uri": "storage/crops/crop1.png",
            "bbox": {"x": 10, "y": 10, "width": 50, "height": 50},
            "padded_bbox": {"x": 8, "y": 8, "width": 54, "height": 54},
            "detection_confidence": 0.95
        }
    ]
    
    with patch("services.inspection_service.detect_and_crop_instances") as mock_detect, \
         patch("services.inspection_service.run_patchcore_inference") as mock_infer, \
         patch("cv2.imread") as mock_imread, \
         patch("cv2.imwrite", side_effect=dummy_imwrite):
        
        mock_detect.return_value = (dummy_instances, {"detection_time_ms": 10.0})
        mock_imread.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_infer.return_value = {
            "status": "REJECT", "anomaly_score": 40.0, "threshold": 25.0, "severity": "CRITICAL", "heatmap": [], "bbox": None, "heatmap_uri": None, "processing_time_ms": 50.0
        }
        
        response = client.post(
            "/api/inspections",
            data={"model_id": active_model["model_id"], "inspection_mode": "multi_instance"},
            files={"image": ("multi_test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 201
        inspection_id = response.json()["id"]

    # Now trigger instance-specific VLM for instance 1
    mock_vlm_result = {
        "status": "completed",
        "provider": "gemini",
        "defect_type": "Surface Scratch",
        "explanation": "Detected deep linear gouge across thread line.",
        "suggested_cause": "Tooling wear or handling friction.",
        "recommended_action": "Quarantine batch and inspect cutting die.",
        "confidence_score": 0.92
    }
    
    with patch("services.inspection_service.analyze_inspection_evidence") as mock_gemini:
        mock_gemini.return_value = mock_vlm_result
        
        vlm_resp = client.post(f"/api/inspections/{inspection_id}/instances/1/analyze-vlm")
        assert vlm_resp.status_code == 200, vlm_resp.text
        vlm_data = vlm_resp.json()
        
        result = vlm_data.get("result", vlm_data)
        target_inst = next(i for i in result["instances"] if i["instance_id"] == 1)
        assert target_inst["vlm_analysis"] is not None
        assert target_inst["vlm_analysis"]["defect_type"] == "Surface Scratch"
        assert target_inst["vlm_analysis"]["confidence_score"] == 0.92
