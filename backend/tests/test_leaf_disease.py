"""
test_leaf_disease.py — Comprehensive Unit Tests for Phase 1 Leaf Disease Analysis
----------------------------------------------------------------------------------
Tests image validation, crop identification, disease provider abstraction,
diagnosis evidence validation, FastAPI endpoint, and CRITICAL GEMINI ISOLATION.
"""

import io
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from fastapi.testclient import TestClient

from api_server import app
from services.leaf_disease.image_validator import validate_single_image, validate_multiple_images
from services.leaf_disease.plant_identifier import identify_crop
from services.leaf_disease.disease_provider import (
    MockDiseaseDetectionProvider,
    PlantDiseaseAPIProvider,
    PlantNetDiseaseProvider,
    get_disease_provider
)
from services.leaf_disease.diagnosis_validator import validate_diagnosis_evidence
from schemas.leaf_disease import (
    OverallImageValidationResult,
    CropIdentificationResult,
    NormalizedDiseaseResult,
    ImageValidationDetail
)


def create_dummy_image_bytes(width=300, height=300, color=(0, 150, 0), fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestLeafDiseasePhase1(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.valid_jpg_bytes = create_dummy_image_bytes(300, 300, fmt="JPEG")

    # -------------------------------------------------------------
    # 1. Image Validation Tests
    # -------------------------------------------------------------
    def test_image_validation_valid(self):
        detail = validate_single_image("test_leaf.jpg", self.valid_jpg_bytes)
        self.assertTrue(detail.is_valid)
        self.assertEqual(detail.format, "JPEG")
        self.assertEqual(detail.width, 300)
        self.assertEqual(detail.height, 300)
        self.assertEqual(len(detail.errors), 0)

    def test_image_validation_empty(self):
        detail = validate_single_image("empty.jpg", b"")
        self.assertFalse(detail.is_valid)
        self.assertIn("Empty file submitted.", detail.errors)

    def test_image_validation_corrupted(self):
        detail = validate_single_image("bad.jpg", b"Not an image file content")
        self.assertFalse(detail.is_valid)
        self.assertTrue(any("Corrupted" in e or "invalid" in e for e in detail.errors))

    def test_image_validation_low_resolution(self):
        tiny_bytes = create_dummy_image_bytes(50, 50, fmt="JPEG")
        detail = validate_single_image("tiny.jpg", tiny_bytes)
        self.assertFalse(detail.is_valid)
        self.assertTrue(any("resolution" in e.lower() for e in detail.errors))

    def test_image_validation_multiple(self):
        files = [
            ("leaf1.jpg", self.valid_jpg_bytes),
            ("leaf2.jpg", self.valid_jpg_bytes),
            ("bad.jpg", b"corrupted content")
        ]
        result = validate_multiple_images(files)
        self.assertEqual(result.total_submitted, 3)
        self.assertEqual(result.valid_count, 2)
        self.assertEqual(result.invalid_count, 1)
        self.assertTrue(result.is_any_valid)

    # -------------------------------------------------------------
    # 2. Crop Identification Tests
    # -------------------------------------------------------------
    def test_crop_identification_user_override(self):
        res = identify_crop(selected_crop="Rice / Paddy")
        self.assertEqual(res.crop_name, "Rice / Paddy")
        self.assertEqual(res.source, "user_selected")
        self.assertEqual(res.status, "USER_SPECIFIED")

    def test_crop_identification_automatic(self):
        res = identify_crop(selected_crop=None, valid_filenames=["tomato_leaf.jpg"])
        self.assertEqual(res.crop_name, "Tomato")
        self.assertEqual(res.source, "automatic")
        self.assertEqual(res.status, "CONFIRMED")

    # -------------------------------------------------------------
    # 3. Disease Provider Abstraction Tests
    # -------------------------------------------------------------
    def test_mock_disease_provider(self):
        provider = MockDiseaseDetectionProvider()
        img_detail = validate_single_image("blight_sample.jpg", self.valid_jpg_bytes)
        res = provider.detect_disease("Tomato", [img_detail])
        self.assertTrue(res.is_mock)
        self.assertEqual(res.provider_name, "MockDiseaseDetectionProvider")
        self.assertEqual(res.provider_status, "MOCK")
        self.assertIn("Blight", res.disease_name)
        self.assertIsNotNone(res.confidence)

    def test_api_provider_unconfigured_key(self):
        provider = PlantDiseaseAPIProvider(api_key="")
        img_detail = validate_single_image("leaf.jpg", self.valid_jpg_bytes)
        res = provider.detect_disease("Tomato", [img_detail])
        self.assertFalse(res.is_mock)
        self.assertEqual(res.provider_status, "PROVIDER_ERROR")
        self.assertIn("PLANT_API_KEY environment variable is not configured", res.error_message or "")

    def test_plantnet_disease_provider_unconfigured_key(self):
        provider = PlantNetDiseaseProvider(api_key="")
        img_detail = validate_single_image("leaf.jpg", self.valid_jpg_bytes)
        res = provider.detect_disease("Tomato", [img_detail])
        self.assertFalse(res.is_mock)
        self.assertEqual(res.provider_status, "PROVIDER_ERROR")
        self.assertEqual(res.provider_name, "Pl@ntNet Diseases API")
        self.assertIn("PLANT_API_KEY environment variable is not configured", res.error_message or "")

    # -------------------------------------------------------------
    # 4. Diagnosis Evidence Validation Tests
    # -------------------------------------------------------------
    def test_diagnosis_evidence_high(self):
        val_res = OverallImageValidationResult(
            total_submitted=1,
            valid_count=1,
            invalid_count=0,
            is_any_valid=True,
            details=[
                ImageValidationDetail(
                    filename="leaf.jpg",
                    is_valid=True,
                    format="JPEG",
                    width=300,
                    height=300,
                    size_bytes=1000,
                    blur_score=150.0,
                    brightness=120.0,
                    errors=[],
                    warnings=[]
                )
            ]
        )
        crop_res = CropIdentificationResult(
            crop_name="Tomato", confidence=0.9, source="automatic", status="CONFIRMED", evidence_note="Confirmed"
        )
        disease_res = NormalizedDiseaseResult(
            disease_name="Tomato Early Blight", confidence=0.92, provider_name="MockProvider", is_mock=True, supported=True, candidates=[], provider_status="MOCK"
        )
        diag = validate_diagnosis_evidence(val_res, crop_res, disease_res)
        self.assertEqual(diag.evidence_level, "HIGH")
        self.assertFalse(diag.is_uncertain)

    def test_diagnosis_evidence_low_on_provider_error(self):
        val_res = validate_multiple_images([("leaf.jpg", self.valid_jpg_bytes)])
        crop_res = CropIdentificationResult(
            crop_name="Tomato", confidence=0.9, source="automatic", status="CONFIRMED", evidence_note="Confirmed"
        )
        disease_res = NormalizedDiseaseResult(
            disease_name="Error", confidence=None, provider_name="PlantDiseaseAPIProvider", is_mock=False, supported=False, candidates=[], provider_status="PROVIDER_ERROR", error_message="Error"
        )
        diag = validate_diagnosis_evidence(val_res, crop_res, disease_res)
        self.assertEqual(diag.evidence_level, "LOW")
        self.assertTrue(diag.is_uncertain)

    def test_kindwise_provider_unconfigured_key(self):
        from services.leaf_disease.disease_provider import KindwiseCropHealthProvider
        provider = KindwiseCropHealthProvider(api_key="")
        img_detail = validate_single_image("leaf.jpg", self.valid_jpg_bytes)
        res = provider.detect_disease("Tomato", [img_detail])
        self.assertFalse(res.is_mock)
        self.assertEqual(res.provider_status, "PROVIDER_ERROR")
        self.assertIn("CROP_HEALTH_API_KEY environment variable is not configured", res.error_message or "")

    # -------------------------------------------------------------
    # 5. FastAPI Endpoint Integration Tests
    # -------------------------------------------------------------
    @patch("api.leaf_disease_router.get_disease_provider")
    def test_api_analyze_success(self, mock_get_provider):
        mock_get_provider.return_value = MockDiseaseDetectionProvider()
        response = self.client.post(
            "/api/leaf-disease/analyze",
            files={"files": ("tomato_blight.jpg", self.valid_jpg_bytes, "image/jpeg")},
            data={"selected_crop": "Tomato"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("analysis_id", data)
        self.assertIn(data["status"], ["SUCCESS", "UNCERTAIN"])
        self.assertEqual(data["crop"]["crop_name"], "Tomato")
        self.assertIsNotNone(data["disease"])
        self.assertIsNotNone(data["diagnosis"])

    def test_api_analyze_no_files_bad_request(self):
        response = self.client.post("/api/leaf-disease/analyze")
        self.assertEqual(response.status_code, 422)  # Missing required form files

    # -------------------------------------------------------------
    # 6. CRITICAL GUARD: ZERO GEMINI CALLS IN PHASE 1
    # -------------------------------------------------------------
    @patch("api.leaf_disease_router.get_disease_provider")
    def test_phase1_makes_zero_gemini_calls(self, mock_get_provider):
        mock_get_provider.return_value = MockDiseaseDetectionProvider()
        response = self.client.post(
            "/api/leaf-disease/analyze",
            files={"files": ("test_leaf.jpg", self.valid_jpg_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("Phase 3", data.get("phase_notice", ""))
        self.assertIn("nacl_recommendations", data)



if __name__ == "__main__":
    unittest.main()
