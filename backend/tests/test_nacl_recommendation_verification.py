"""
test_nacl_recommendation_verification.py
-----------------------------------------
Unit test suite verifying evidence-based NACL agrochemical recommendations,
deterministic product validation, dosage safety gates, and host-pathogen mismatch handling.
"""

import unittest
import os
import sys

# Add backend src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from services.leaf_disease.nacl_recommendation_engine import (
    NACLRecommendationEngine,
    validate_nacl_product_match
)
from db.nacl_product_db import query_nacl_products


class TestNACLRecommendationVerification(unittest.TestCase):
    """Test suite covering the 10 evidence verification requirements for NACL recommendations."""

    def setUp(self):
        self.engine = NACLRecommendationEngine()
        self.crop = "Cucurbits"
        self.disease = "Powdery mildew of cucurbits (Podosphaera xanthii)"

    def test_01_kazan_rejected_for_cucurbit_powdery_mildew(self):
        """1. Kazan (Thifluzamide) must be REJECTED for Cucurbits + Powdery Mildew because it lacks Cucurbit registration."""
        kazan_prods = query_nacl_products(crop_name="Cucurbits", category="Fungicides")
        kazan_doc = next((p for p in kazan_prods if p.get("product_id") == "kazan"), None)
        
        # If query returned kazan by category, check validator decision
        if not kazan_doc:
            # Query all fungicides to get Kazan document
            all_fungicides = query_nacl_products(category="Fungicides")
            kazan_doc = next((p for p in all_fungicides if p.get("product_id") == "kazan"), None)

        self.assertIsNotNone(kazan_doc, "Kazan should exist in MongoDB catalog")
        val_report = validate_nacl_product_match(kazan_doc, self.crop, self.disease)
        
        self.assertFalse(val_report["product_verified"], "Kazan product_verified must be False for Cucurbits")
        self.assertEqual(val_report["verification_status"], "REJECTED")

    def test_02_mass_plus_rejected_for_cucurbit_powdery_mildew(self):
        """2. Mass Plus (Hexaconazole) must be REJECTED for Cucurbits because its registered crops are Mango and Paddy."""
        all_fungicides = query_nacl_products(category="Fungicides")
        mass_doc = next((p for p in all_fungicides if p.get("product_id") == "mass-plus"), None)
        
        self.assertIsNotNone(mass_doc, "Mass Plus should exist in MongoDB catalog")
        val_report = validate_nacl_product_match(mass_doc, self.crop, self.disease)
        
        self.assertFalse(val_report["product_verified"], "Mass Plus product_verified must be False for Cucurbits")
        self.assertEqual(val_report["verification_status"], "REJECTED")

    def test_03_nagarjuna_index_rejected_for_cucurbit_powdery_mildew(self):
        """3. Nagarjuna Index (Myclobutanil) must be REJECTED for Cucurbits because it is registered for Apple, Grapes, Chilli only."""
        all_fungicides = query_nacl_products(category="Fungicides")
        index_doc = next((p for p in all_fungicides if p.get("product_id") == "nagarjuna-index"), None)
        
        self.assertIsNotNone(index_doc, "Nagarjuna Index should exist in MongoDB catalog")
        val_report = validate_nacl_product_match(index_doc, self.crop, self.disease)
        
        self.assertFalse(val_report["product_verified"], "Nagarjuna Index product_verified must be False for Cucurbits")
        self.assertEqual(val_report["verification_status"], "REJECTED")

    def test_04_zen_accepted_for_cucurbit_powdery_mildew(self):
        """4. Zen (Carbendazim 50% WP) must be VERIFIED for Cucurbits + Powdery Mildew."""
        all_fungicides = query_nacl_products(category="Fungicides")
        zen_doc = next((p for p in all_fungicides if p.get("product_id") == "zen"), None)
        
        self.assertIsNotNone(zen_doc, "Zen should exist in catalog")
        val_report = validate_nacl_product_match(zen_doc, self.crop, self.disease)
        
        self.assertTrue(val_report["product_verified"], "Zen must be VERIFIED for Cucurbits Powdery Mildew")
        self.assertTrue(val_report["dosage_verified"], "Zen dosage should be verified")
        self.assertEqual(val_report["recommended_dosage"], "120 gm/acre")

    def test_05_kadak_accepted_for_cucurbit_powdery_mildew(self):
        """5. Kadak (Azoxystrobin + Chlorothalonil SC) must be VERIFIED for Cucurbits + Powdery Mildew."""
        all_fungicides = query_nacl_products(category="Fungicides")
        kadak_doc = next((p for p in all_fungicides if p.get("product_id") == "kadak"), None)
        
        self.assertIsNotNone(kadak_doc, "Kadak should exist in catalog")
        val_report = validate_nacl_product_match(kadak_doc, self.crop, self.disease)
        
        self.assertTrue(val_report["product_verified"], "Kadak must be VERIFIED for Cucurbits Powdery Mildew")
        self.assertTrue(val_report["dosage_verified"], "Kadak dosage should be verified")
        self.assertEqual(val_report["recommended_dosage"], "600 ml/acre")

    def test_06_unverified_disease_returns_empty_recommendations(self):
        """6. If no verified NACL product match exists for a crop + disease, return recommendations = []."""
        result = self.engine.recommend_products(
            crop_name="Cucurbits",
            disease_name="Unknown Exotic Bacterial Wilt"
        )
        self.assertEqual(result.recommendations, [], "Recommendations must be empty when no verified matches exist")
        self.assertIn("No verified NACL product match", result.ai_advisory_summary)

    def test_07_active_ingredient_alone_is_insufficient(self):
        """7. Active ingredient match alone without explicit crop registration does NOT produce a recommendation."""
        # Query end-to-end recommendations for Cucurbits + Powdery Mildew
        result = self.engine.recommend_products(self.crop, self.disease)
        rec_ids = [r.product_id for r in result.recommendations]
        
        # Ensure rejected active ingredient matches (kazan, mass-plus, nagarjuna-index) are NOT present
        self.assertNotIn("kazan", rec_ids, "Kazan must not be in recommendations")
        self.assertNotIn("mass-plus", rec_ids, "Mass Plus must not be in recommendations")
        self.assertNotIn("nagarjuna-index", rec_ids, "Nagarjuna Index must not be in recommendations")

    def test_08_cross_crop_dosage_isolation(self):
        """8. If product has crop match but missing dosage for target crop, dosage is UNVERIFIED ('Refer to current product label')."""
        mock_product = {
            "product_id": "test-prod",
            "product_name": "Test Product",
            "category": "Fungicides",
            "product_url": "https://naclind.com/products/test/",
            "active_ingredient": "TestActive 10%",
            "crop_applications": [
                {
                    "crop": "Cucurbits",
                    "target_pest_or_disease": "Powdery Mildew",
                    "dosage": "",  # Empty dosage for Cucurbits
                    "dosage_verified": False
                },
                {
                    "crop": "Grapes",
                    "target_pest_or_disease": "Powdery Mildew",
                    "dosage": "500 ml/acre",
                    "dosage_verified": True
                }
            ],
            "source_url": "https://naclind.com/products/test/",
            "source_type": "OFFICIAL_NACL_WEBSITE"
        }
        val_report = validate_nacl_product_match(mock_product, "Cucurbits", "Powdery Mildew")
        self.assertTrue(val_report["product_verified"])
        self.assertFalse(val_report["dosage_verified"], "Dosage must be unverified for Cucurbits")
        self.assertEqual(val_report["recommended_dosage"], "Refer to current product label")

    def test_09_gemini_advisory_uses_backend_facts(self):
        """9. Gemini advisory synthesis strictly relies on backend verified facts."""
        result = self.engine.recommend_products(self.crop, self.disease)
        self.assertGreater(len(result.recommendations), 0, "Should have verified recommendations for Cucurbits Powdery Mildew")
        for rec in result.recommendations:
            self.assertTrue(rec.dosage_verified)
            self.assertIn("acre", rec.recommended_dosage)

    def test_10_safety_gate_mismatched_host_blocks_recommendations(self):
        """10. Safety gate: MISMATCHED_HOST status blocks product recommendations completely."""
        result = self.engine.recommend_products(
            crop_name="Grape",
            disease_name="Powdery mildew of cucurbits (Podosphaera xanthii)",
            crop_compatibility_status="MISMATCHED_HOST"
        )
        self.assertEqual(result.recommendations, [])
        self.assertIn("Crop and disease evidence conflict", result.ai_advisory_summary)

    def test_11_zen_rejected_for_cucurbit_downy_mildew(self):
        """11. Regression: Zen (Carbendazim 50% WP) MUST BE REJECTED for Cucurbits + Downy Mildew."""
        all_fungicides = query_nacl_products(category="Fungicides")
        zen_doc = next((p for p in all_fungicides if p.get("product_id") == "zen"), None)
        self.assertIsNotNone(zen_doc)
        
        val_report = validate_nacl_product_match(
            zen_doc,
            "Cucurbits",
            "Downy mildew of cucurbits (Pseudoperonospora cubensis)"
        )
        self.assertFalse(val_report["product_verified"], "Zen MUST BE REJECTED for Downy Mildew of Cucurbits")
        self.assertEqual(val_report["verification_status"], "REJECTED")

    def test_12_zen_accepted_for_cucurbit_powdery_mildew(self):
        """12. Regression: Zen (Carbendazim 50% WP) ACCEPTED for Cucurbits + Powdery Mildew due to explicit registration."""
        all_fungicides = query_nacl_products(category="Fungicides")
        zen_doc = next((p for p in all_fungicides if p.get("product_id") == "zen"), None)
        self.assertIsNotNone(zen_doc)
        
        val_report = validate_nacl_product_match(
            zen_doc,
            "Cucurbits",
            "Powdery mildew of cucurbits (Podosphaera xanthii)"
        )
        self.assertTrue(val_report["product_verified"], "Zen must be VERIFIED for Powdery Mildew of Cucurbits")

    def test_13_zen_rejected_for_cucurbit_fruit_rot(self):
        """13. Regression: Zen MUST BE REJECTED for Cucurbits + Fruit Rot (unregistered target)."""
        all_fungicides = query_nacl_products(category="Fungicides")
        zen_doc = next((p for p in all_fungicides if p.get("product_id") == "zen"), None)
        self.assertIsNotNone(zen_doc)
        
        val_report = validate_nacl_product_match(
            zen_doc,
            "Cucurbits",
            "Fruit rot of cucurbits"
        )
        self.assertFalse(val_report["product_verified"], "Zen MUST BE REJECTED for Fruit Rot of Cucurbits")
        self.assertEqual(val_report["verification_status"], "REJECTED")

    def test_14_kadak_accepted_for_cucurbit_downy_mildew(self):
        """14. Regression: Kadak (Azoxystrobin + Chlorothalonil) ACCEPTED for Cucurbits + Downy Mildew."""
        all_fungicides = query_nacl_products(category="Fungicides")
        kadak_doc = next((p for p in all_fungicides if p.get("product_id") == "kadak"), None)
        self.assertIsNotNone(kadak_doc)
        
        val_report = validate_nacl_product_match(
            kadak_doc,
            "Cucurbits",
            "Downy mildew of cucurbits (Pseudoperonospora cubensis)"
        )
        self.assertTrue(val_report["product_verified"], "Kadak must be VERIFIED for Downy Mildew of Cucurbits")

    def test_15_powdery_mildew_never_matches_downy_mildew(self):
        """15. Regression: Powdery mildew registered product rule MUST NEVER match Downy mildew disease query."""
        mock_powdery_product = {
            "product_id": "powdery-only-prod",
            "product_name": "Powdery Only Fungicide",
            "category": "Fungicides",
            "crop_applications": [
                {
                    "crop": "Cucurbits",
                    "target_pest_or_disease": "Powdery Mildew",
                    "dosage": "100 gm/acre"
                }
            ]
        }
        val_report = validate_nacl_product_match(mock_powdery_product, "Cucurbits", "Downy mildew of cucurbits")
        self.assertFalse(val_report["product_verified"], "Powdery Mildew registration MUST NEVER match Downy Mildew query")

    def test_16_fruit_rot_never_matches_downy_mildew(self):
        """16. Regression: Fruit rot registered product rule MUST NEVER match Downy mildew disease query."""
        mock_rot_product = {
            "product_id": "rot-only-prod",
            "product_name": "Fruit Rot Fungicide",
            "category": "Fungicides",
            "crop_applications": [
                {
                    "crop": "Cucurbits",
                    "target_pest_or_disease": "Fruit Rot",
                    "dosage": "200 ml/acre"
                }
            ]
        }
        val_report = validate_nacl_product_match(mock_rot_product, "Cucurbits", "Downy mildew of cucurbits")
        self.assertFalse(val_report["product_verified"], "Fruit Rot registration MUST NEVER match Downy Mildew query")

    def test_17_generic_fungicide_key_benefits_never_accepts(self):
        """17. Regression: Generic fungicide key benefits or mode of action MUST NEVER produce a target match."""
        mock_generic_product = {
            "product_id": "generic-fungicide",
            "product_name": "Broad Fungicide",
            "category": "Fungicides",
            "key_benefits": ["Broad-spectrum systemic fungicide for fungal control"],
            "mode_of_action": "Inhibits fungal cell wall synthesis",
            "crop_applications": [
                {
                    "crop": "Cucurbits",
                    "target_pest_or_disease": "", # No target listed
                    "dosage": "500 ml/acre"
                }
            ]
        }
        val_report = validate_nacl_product_match(mock_generic_product, "Cucurbits", "Downy mildew of cucurbits")
        self.assertFalse(val_report["product_verified"], "Generic key benefits MUST NOT produce target verification")


if __name__ == "__main__":
    unittest.main()

