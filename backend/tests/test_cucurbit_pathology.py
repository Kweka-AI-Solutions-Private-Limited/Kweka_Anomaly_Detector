"""
test_cucurbit_pathology.py — Unit Tests for Cucurbit Pathology & Host Compatibility
--------------------------------------------------------------------------------------
Validates:
1. Cucurbits is recognized as a supported crop.
2. Podosphaera xanthii -> Cucurbits host mapping.
3. Erysiphe necator -> Grape host mapping.
4. Grape + Podosphaera xanthii -> MISMATCHED_HOST.
5. MISMATCHED_HOST -> LOW evidence.
6. MISMATCHED_HOST -> UNCERTAIN diagnosis.
7. MISMATCHED_HOST -> NO NACL recommendation.
8. Grape + Erysiphe necator -> compatible.
9. Confirmed Cucurbits + Podosphaera xanthii -> compatible.
10. System NEVER silently changes Grape to Cucurbits merely from the pathogen.
11. No dosage is generated when crop identity is unresolved.
"""

import unittest
from schemas.leaf_disease import (
    OverallImageValidationResult,
    ImageValidationDetail,
    CropIdentificationResult,
    NormalizedDiseaseResult,
)
from services.leaf_disease.plant_identifier import identify_crop, KNOWN_CROPS
from services.leaf_disease.diagnosis_validator import validate_diagnosis_evidence, _check_host_compatibility, PATHOGEN_HOST_MAP
from services.leaf_disease.nacl_recommendation_engine import NACLRecommendationEngine


class TestCucurbitPathology(unittest.TestCase):

    def setUp(self):
        self.engine = NACLRecommendationEngine()
        self.dummy_validation = OverallImageValidationResult(
            total_submitted=1,
            valid_count=1,
            invalid_count=0,
            is_any_valid=True,
            details=[
                ImageValidationDetail(
                    filename="leaf.jpg",
                    is_valid=True,
                    width=1024,
                    height=768,
                    blur_score=50.0,
                    brightness=120.0,
                    errors=[],
                    warnings=[]
                )
            ]
        )

    def test_1_cucurbits_recognized_as_supported_crop(self):
        """1. Cucurbits is recognized as a supported crop."""
        self.assertIn("cucurbits", KNOWN_CROPS)
        self.assertIn("cucumber", KNOWN_CROPS)
        self.assertIn("melon", KNOWN_CROPS)
        res = identify_crop(selected_crop="Cucurbits")
        self.assertEqual(res.crop_name, "Cucurbits")
        self.assertEqual(res.status, "USER_SPECIFIED")

    def test_2_podosphaera_xanthii_host_mapping(self):
        """2. Podosphaera xanthii -> Cucurbits host mapping."""
        is_compat = _check_host_compatibility(crop_name="Cucurbits", disease_name="Powdery mildew of cucurbits (Podosphaera xanthii)")
        self.assertTrue(is_compat)

    def test_3_erysiphe_necator_host_mapping(self):
        """3. Erysiphe necator -> Grape host mapping."""
        is_compat = _check_host_compatibility(crop_name="Grape", disease_name="Grape Powdery Mildew (Erysiphe necator)")
        self.assertTrue(is_compat)

    def test_4_grape_plus_podosphaera_xanthii_mismatched_host(self):
        """4. Grape + Podosphaera xanthii -> MISMATCHED_HOST."""
        crop_res = CropIdentificationResult(crop_name="Grape", confidence=0.85, source="automatic", status="CONFIRMED", evidence_note="test evidence")
        disease_res = NormalizedDiseaseResult(
            disease_name="Powdery mildew of cucurbits (Podosphaera xanthii)",
            confidence=0.92,
            provider_name="Pl@ntNet",
            supported=True,
            provider_status="OK"
        )
        val_res = validate_diagnosis_evidence(self.dummy_validation, crop_res, disease_res)
        self.assertEqual(val_res.crop_compatibility_status, "MISMATCHED_HOST")

    def test_5_mismatched_host_demotes_to_low_evidence(self):
        """5. MISMATCHED_HOST -> LOW evidence."""
        crop_res = CropIdentificationResult(crop_name="Grape", confidence=0.90, source="automatic", status="CONFIRMED", evidence_note="test evidence")
        disease_res = NormalizedDiseaseResult(
            disease_name="Powdery mildew of cucurbits (Podosphaera xanthii)",
            confidence=0.95,
            provider_name="Pl@ntNet",
            supported=True,
            provider_status="OK"
        )
        val_res = validate_diagnosis_evidence(self.dummy_validation, crop_res, disease_res)
        self.assertEqual(val_res.evidence_level, "LOW")

    def test_6_mismatched_host_results_in_uncertain_diagnosis(self):
        """6. MISMATCHED_HOST -> UNCERTAIN diagnosis."""
        crop_res = CropIdentificationResult(crop_name="Grape", confidence=0.90, source="automatic", status="CONFIRMED", evidence_note="test evidence")
        disease_res = NormalizedDiseaseResult(
            disease_name="Powdery mildew of cucurbits (Podosphaera xanthii)",
            confidence=0.95,
            provider_name="Pl@ntNet",
            supported=True,
            provider_status="OK"
        )
        val_res = validate_diagnosis_evidence(self.dummy_validation, crop_res, disease_res)
        self.assertTrue(val_res.is_uncertain)

    def test_7_mismatched_host_blocks_nacl_recommendations(self):
        """7. MISMATCHED_HOST -> NO NACL recommendation."""
        rec = self.engine.recommend_products(
            crop_name="Grape",
            disease_name="Powdery mildew of cucurbits (Podosphaera xanthii)",
            crop_compatibility_status="MISMATCHED_HOST",
            is_uncertain=True
        )
        self.assertEqual(len(rec.recommendations), 0)
        self.assertEqual(len(rec.recommended_active_ingredients), 0)
        self.assertIn("conflict", rec.ai_advisory_summary.lower())

    def test_8_grape_plus_erysiphe_necator_compatible(self):
        """8. Grape + Erysiphe necator -> compatible."""
        crop_res = CropIdentificationResult(crop_name="Grape", confidence=0.90, source="automatic", status="CONFIRMED", evidence_note="test evidence")
        disease_res = NormalizedDiseaseResult(
            disease_name="Grape Powdery Mildew (Erysiphe necator)",
            confidence=0.90,
            provider_name="Pl@ntNet",
            supported=True,
            provider_status="OK"
        )
        val_res = validate_diagnosis_evidence(self.dummy_validation, crop_res, disease_res)
        self.assertEqual(val_res.crop_compatibility_status, "COMPATIBLE")
        self.assertFalse(val_res.is_uncertain)

    def test_9_confirmed_cucurbits_plus_podosphaera_xanthii_compatible(self):
        """9. Confirmed Cucurbits + Podosphaera xanthii -> compatible."""
        crop_res = CropIdentificationResult(crop_name="Cucurbits", confidence=0.90, source="automatic", status="CONFIRMED", evidence_note="test evidence")
        disease_res = NormalizedDiseaseResult(
            disease_name="Powdery mildew of cucurbits (Podosphaera xanthii)",
            confidence=0.90,
            provider_name="Pl@ntNet",
            supported=True,
            provider_status="OK"
        )
        val_res = validate_diagnosis_evidence(self.dummy_validation, crop_res, disease_res)
        self.assertEqual(val_res.crop_compatibility_status, "COMPATIBLE")
        self.assertFalse(val_res.is_uncertain)

    def test_10_system_never_silently_changes_crop(self):
        """10. System NEVER silently changes Grape to Cucurbits merely from the pathogen."""
        crop_res = CropIdentificationResult(crop_name="Grape", confidence=0.85, source="automatic", status="CONFIRMED", evidence_note="test evidence")
        disease_res = NormalizedDiseaseResult(
            disease_name="Powdery mildew of cucurbits (Podosphaera xanthii)",
            confidence=0.95,
            provider_name="Pl@ntNet",
            supported=True,
            provider_status="OK"
        )
        val_res = validate_diagnosis_evidence(self.dummy_validation, crop_res, disease_res)
        # Crop identification remains Grape, host compatibility status is MISMATCHED_HOST
        self.assertEqual(crop_res.crop_name, "Grape")
        self.assertEqual(val_res.crop_compatibility_status, "MISMATCHED_HOST")

    def test_11_no_dosage_generated_when_crop_unresolved(self):
        """11. No dosage is generated when crop identity is unresolved."""
        rec = self.engine.recommend_products(
            crop_name="Unknown",
            disease_name="Powdery mildew of cucurbits (Podosphaera xanthii)",
            is_uncertain=True
        )
        self.assertEqual(len(rec.recommendations), 0)
        for r in rec.recommendations:
            self.assertIsNone(r.recommended_dosage)


if __name__ == "__main__":
    unittest.main()
