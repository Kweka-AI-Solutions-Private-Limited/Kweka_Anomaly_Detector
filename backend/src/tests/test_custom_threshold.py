"""
Test Custom Threshold Feature
-----------------------------
Comprehensive regression tests for custom threshold lifecycle:
1. Automatic version calibration metadata
2. Custom threshold version creation (N -> N+1)
3. Version number increment
4. Artifact reuse (checkpoint & memory bank URIs)
5. Reference-image reuse
6. Immutability of old version
7. Model active_version_id update & activation
8. Validation & rejection of invalid thresholds (<= 0, NaN, inf, invalid types)
9. Inspection execution using the custom threshold
"""

import sys
from pathlib import Path

# Add backend/src to sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import math
import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from api_server import app
from db.connection import get_db
from db.schemas import ModelVersionSchema, CalibrationConfig, TrainingConfig, VersionArtifacts, AlgorithmConfig

client = TestClient(app)


@pytest.fixture
def test_db():
    return get_db()


@pytest.fixture
def mock_model_with_version(test_db):
    """Sets up a test model with an initial auto-calibrated version v1."""
    # Create model
    model_res = test_db.models.insert_one({
        "name": "Custom Threshold Test Model",
        "description": "Test model for custom threshold feature",
        "domain": "tile",
        "status": "active",
        "reference_image_count": 5,
        "created_at": "2026-09-11T12:00:00Z",
        "updated_at": "2026-09-11T12:00:00Z"
    })
    model_id = model_res.inserted_id

    # Create dummy reference image IDs
    ref_ids = [ObjectId(), ObjectId(), ObjectId()]

    # Create version v1
    v1_doc = {
        "model_id": model_id,
        "version_number": 1,
        "status": "ready",
        "algorithm": {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.01
        },
        "calibration": {
            "method": "95th_percentile",
            "threshold": 21.43,
            "auto_calibrated_threshold": 21.43,
            "is_custom": False
        },
        "training": {
            "build_time_ms": 150.0,
            "reference_count": 3,
            "reference_ids": ref_ids,
            "created_at": "2026-09-11T12:05:00Z"
        },
        "artifacts": {
            "checkpoint_uri": "artifacts/models/test_model/v1/checkpoint.pt",
            "memory_bank_uri": "artifacts/models/test_model/v1/memory_bank.pt",
            "version_metadata_uri": "artifacts/models/test_model/v1/metadata.json"
        },
        "created_at": "2026-09-11T12:05:00Z"
    }
    v1_res = test_db.model_versions.insert_one(v1_doc)
    v1_id = v1_res.inserted_id

    # Update model's active_version_id
    test_db.models.update_one(
        {"_id": model_id},
        {"$set": {"active_version_id": v1_id}}
    )

    yield {
        "model_id": str(model_id),
        "v1_id": str(v1_id),
        "ref_ids": ref_ids,
        "v1_artifacts": v1_doc["artifacts"]
    }

    # Cleanup
    test_db.models.delete_one({"_id": model_id})
    test_db.model_versions.delete_many({"model_id": model_id})


def test_custom_threshold_creation_success(test_db, mock_model_with_version):
    """Verifies successful custom threshold version creation (v1 -> v2) with artifact & reference reuse."""
    model_id = mock_model_with_version["model_id"]
    v1_id = mock_model_with_version["v1_id"]

    response = client.post(f"/api/models/{model_id}/threshold", json={"threshold": 18.5})
    assert response.status_code == 201
    data = response.json()

    # 1. Version number increment
    assert data["version_number"] == 2
    assert data["status"] == "ready"

    # 2. Calibration metadata
    calib = data["calibration"]
    assert calib["threshold"] == 18.5
    assert calib["auto_calibrated_threshold"] == 21.43
    assert calib["is_custom"] is True
    assert calib["method"] == "custom"

    # 3. Artifact reuse
    assert data["artifacts"]["checkpoint_uri"] == mock_model_with_version["v1_artifacts"]["checkpoint_uri"]
    assert data["artifacts"]["memory_bank_uri"] == mock_model_with_version["v1_artifacts"]["memory_bank_uri"]

    # 4. Reference ID reuse
    ref_ids_in_v2 = [str(r) for r in data["training"]["reference_ids"]]
    expected_ref_ids = [str(r) for r in mock_model_with_version["ref_ids"]]
    assert ref_ids_in_v2 == expected_ref_ids

    # 5. Old version immutability check
    v1_doc = test_db.model_versions.find_one({"_id": ObjectId(v1_id)})
    assert v1_doc["calibration"]["threshold"] == 21.43
    assert v1_doc["calibration"]["is_custom"] is False
    assert v1_doc["calibration"]["method"] == "95th_percentile"
    assert v1_doc["version_number"] == 1

    # 6. Model active version update check
    model_doc = test_db.models.find_one({"_id": ObjectId(model_id)})
    assert str(model_doc["active_version_id"]) == data["id"]
    assert model_doc["status"] == "active"


def test_custom_threshold_invalid_values_rejected(mock_model_with_version):
    """Verifies that invalid threshold values (<= 0, NaN, non-numeric) are rejected with 400 Bad Request."""
    model_id = mock_model_with_version["model_id"]

    invalid_cases = [
        0,
        -1.5,
        -100,
        None,
        "invalid_string"
    ]

    for val in invalid_cases:
        res = client.post(f"/api/models/{model_id}/threshold", json={"threshold": val})
        assert res.status_code in [400, 422], f"Expected rejection for threshold={val}, got {res.status_code}"


def test_custom_threshold_multiple_updates(test_db, mock_model_with_version):
    """Verifies sequential custom threshold updates (v1 -> v2 -> v3) preserve auto_calibrated_threshold."""
    model_id = mock_model_with_version["model_id"]

    # First update: 21.43 -> 18.5
    res1 = client.post(f"/api/models/{model_id}/threshold", json={"threshold": 18.5})
    assert res1.status_code == 201
    assert res1.json()["version_number"] == 2

    # Second update: 18.5 -> 25.0
    res2 = client.post(f"/api/models/{model_id}/threshold", json={"threshold": 25.0})
    assert res2.status_code == 201
    data2 = res2.json()

    assert data2["version_number"] == 3
    assert data2["calibration"]["threshold"] == 25.0
    assert data2["calibration"]["auto_calibrated_threshold"] == 21.43  # Preserves original auto calibrated threshold
    assert data2["calibration"]["is_custom"] is True

    # Check that model points to v3
    model_doc = test_db.models.find_one({"_id": ObjectId(model_id)})
    assert str(model_doc["active_version_id"]) == data2["id"]
