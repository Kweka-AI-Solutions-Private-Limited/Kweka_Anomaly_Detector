"""
Unit & Integration Tests for InspectAI Gemini VLM Service
-----------------------------------------------------------
Verifies VLM analysis schema, fallback handling, inspection service integration,
and retry API endpoint.
"""

import os
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fastapi.testclient import TestClient
from db.schemas import VLMAnalysisSchema, InspectionResultSchema
from services.vlm_service import analyze_inspection_evidence, get_gemini_api_key
from api_server import app

client = TestClient(app)


def test_vlm_schema_structure():
    """Verifies VLMAnalysisSchema fields and defaults."""
    vlm = VLMAnalysisSchema(
        status="completed",
        provider="gemini",
        model="gemini-2.5-flash",
        defect_type="Scratch",
        location="Upper right edge",
        severity="High",
        prominence="Moderate",
        explanation="Visible surface scratch.",
        visual_evidence="Discontinuity detected.",
        confidence=0.92
    )
    dumped = vlm.model_dump()
    assert dumped["status"] == "completed"
    assert dumped["defect_type"] == "Scratch"
    assert dumped["severity"] == "High"
    assert dumped["confidence"] == 0.92


def test_vlm_unavailable_fallback():
    """Verifies analyze_inspection_evidence returns status='unavailable' when API key is missing."""
    with patch("services.vlm_service.get_gemini_api_key", return_value=None):
        res = analyze_inspection_evidence(
            original_image_path="non_existent.png",
            anomaly_score=35.0,
            threshold=25.0
        )
        assert res["status"] == "unavailable"
        assert res["provider"] == "gemini"
        assert "GEMINI_API_KEY" in res["explanation"]


@patch("services.vlm_service.get_gemini_api_key", return_value="mock_api_key")
def test_vlm_mocked_gemini_response(mock_key):
    """Verifies Gemini response parsing and conversion into VLMAnalysisSchema."""
    mock_json_str = '{"defect_type": "Missing Component", "location": "Center-left C12", "severity": "Critical", "prominence": "High", "explanation": "Component C12 is missing from PCB.", "visual_evidence": "Empty pad space in heatmap.", "confidence": 0.95}'

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = mock_json_str
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        res = analyze_inspection_evidence(
            original_image_path="non_existent.png",
            anomaly_score=35.0,
            threshold=25.0
        )
        assert res["status"] == "completed"
        assert res["defect_type"] == "Missing Component"
        assert res["location"] == "Center-left C12"
        assert res["severity"] == "Critical"
        assert res["confidence"] == 0.95
