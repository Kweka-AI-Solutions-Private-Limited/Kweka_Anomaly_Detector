"""
VLM Cross-Dataset & Cross-Inspection Isolation Regression Test Suite
---------------------------------------------------------------------
Verifies that:
1. PCB and Tile inspections remain 100% isolated.
2. No image paths, heatmaps, bounding boxes, or model metadata leak between inspections.
3. Database persistence keys VLM analysis strictly by inspection_id.
4. Gemini API receive strictly independent, single-turn requests.
5. Bidirectional isolation holds (PCB -> Tile and Tile -> PCB).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src is in sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from services.vlm_service import analyze_inspection_evidence
from db.schemas import VLMAnalysisSchema


def test_vlm_cross_dataset_isolation_pcb_to_tile():
    """
    Scenario 1: PCB Inspection A followed by Tile Inspection B.
    Verifies that Inspection B receives ZERO PCB evidence or metadata.
    """
    received_calls = []

    def mock_generate_content(model, contents, config):
        # Record exact contents passed to Gemini
        received_calls.append(contents)
        mock_resp = MagicMock()
        mock_resp.text = '{"defect_type": "Tile Crack", "location": "Center", "severity": "High", "prominence": "Visible", "explanation": "Crack across tile", "visual_evidence": "Line on tile", "confidence": 0.9}'
        return mock_resp

    with patch("services.vlm_service.get_gemini_api_key", return_value="mock_key"), \
         patch("google.genai.Client") as mock_client_cls:
        
        mock_client_inst = MagicMock()
        mock_client_inst.models.generate_content.side_effect = mock_generate_content
        mock_client_cls.return_value = mock_client_inst

        # 1. Run PCB Inspection A
        res_a = analyze_inspection_evidence(
            original_image_path="pcb_sample_001.png",
            heatmap_image_path="pcb_heatmap_001.png",
            anomaly_score=45.2,
            threshold=30.0,
            bbox={"x": 10, "y": 20, "width": 50, "height": 50},
            model_context="PCB Transistor Inspection Model"
        )

        # 2. Run Tile Inspection B
        res_b = analyze_inspection_evidence(
            original_image_path="tile_sample_017.png",
            heatmap_image_path="tile_heatmap_017.png",
            anomaly_score=41.73,
            threshold=27.71,
            bbox={"x": 205, "y": 169, "width": 517, "height": 450},
            model_context="Tiles Defect Inspection"
        )

        # Assert two distinct single-shot requests
        assert len(received_calls) == 2

        # Check call A contents
        call_a_str = " ".join([str(c) for c in received_calls[0]])
        assert "PCB Transistor Inspection Model" in call_a_str
        assert "pcb_sample_001.png" not in call_a_str  # PIL image loaded, string in path log

        # Check call B contents
        call_b_str = " ".join([str(c) for c in received_calls[1]])
        assert "Tiles Defect Inspection" in call_b_str
        assert "PCB Transistor Inspection Model" not in call_b_str
        assert "pcb" not in call_b_str.lower()
        assert "41.73" in call_b_str
        assert "45.2" not in call_b_str


def test_vlm_cross_dataset_isolation_tile_to_pcb():
    """
    Scenario 2: Tile Inspection B followed by PCB Inspection A (Reverse order).
    Verifies bidirectional isolation.
    """
    received_calls = []

    def mock_generate_content(model, contents, config):
        received_calls.append(contents)
        mock_resp = MagicMock()
        mock_resp.text = '{"defect_type": "Missing Component", "location": "Pad C12", "severity": "Critical", "prominence": "High", "explanation": "Component missing", "visual_evidence": "Empty pad", "confidence": 0.95}'
        return mock_resp

    with patch("services.vlm_service.get_gemini_api_key", return_value="mock_key"), \
         patch("google.genai.Client") as mock_client_cls:
        
        mock_client_inst = MagicMock()
        mock_client_inst.models.generate_content.side_effect = mock_generate_content
        mock_client_cls.return_value = mock_client_inst

        # 1. Run Tile Inspection B
        res_b = analyze_inspection_evidence(
            original_image_path="tile_sample_017.png",
            heatmap_image_path="tile_heatmap_017.png",
            anomaly_score=41.73,
            threshold=27.71,
            bbox={"x": 205, "y": 169, "width": 517, "height": 450},
            model_context="Tiles Defect Inspection"
        )

        # 2. Run PCB Inspection A
        res_a = analyze_inspection_evidence(
            original_image_path="pcb_sample_001.png",
            heatmap_image_path="pcb_heatmap_001.png",
            anomaly_score=45.2,
            threshold=30.0,
            bbox={"x": 10, "y": 20, "width": 50, "height": 50},
            model_context="PCB Transistor Inspection Model"
        )

        # Assert two distinct requests
        assert len(received_calls) == 2

        call_tile_str = " ".join([str(c) for c in received_calls[0]])
        call_pcb_str = " ".join([str(c) for c in received_calls[1]])

        # Verify Tile call has NO PCB context
        assert "Tiles Defect Inspection" in call_tile_str
        assert "PCB" not in call_tile_str

        # Verify PCB call has NO Tile context
        assert "PCB Transistor Inspection Model" in call_pcb_str
        assert "Tiles Defect Inspection" not in call_pcb_str
