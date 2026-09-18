"""
test_nacl_recommendation.py — Unit Tests for Phase 3 NACL Recommendation Engine
----------------------------------------------------------------------------------
"""

import unittest
from services.leaf_disease.nacl_recommendation_engine import recommendation_engine
from schemas.nacl_catalog import ProductRecommendationResult


class TestNACLRecommendationEngine(unittest.TestCase):
    """Test suite covering Phase 3 NACL product recommendation logic."""

    def test_healthy_plant_recommendation(self):
        """Verify healthy plants return zero product recommendations."""
        res = recommendation_engine.recommend_products(
            crop_name="Pear",
            disease_name="Healthy Leaf",
            is_healthy=True
        )
        self.assertIsInstance(res, ProductRecommendationResult)
        self.assertTrue(res.is_healthy)
        self.assertEqual(len(res.recommendations), 0)
        self.assertIn("healthy", res.ai_advisory_summary.lower())

    def test_fungal_disease_recommendation(self):
        """Verify fungal disease (Grape Powdery Mildew) returns verified fungicide recommendations."""
        res = recommendation_engine.recommend_products(
            crop_name="Grapes",
            disease_name="Powdery mildew of grape (Uncinula necator)",
            is_healthy=False
        )
        self.assertIsInstance(res, ProductRecommendationResult)
        self.assertFalse(res.is_healthy)
        self.assertGreater(len(res.recommendations), 0, "Should return verified NACL fungicide recommendations")
        
        # Verify category is Fungicides
        categories = [r.category for r in res.recommendations]
        self.assertIn("Fungicides", categories)

        # Verify verification status
        for rec in res.recommendations:
            self.assertTrue(rec.verification_status.startswith("VERIFIED"))

    def test_insect_pest_recommendation(self):
        """Verify insect pest (Cotton Aphids) returns verified insecticide recommendations."""
        res = recommendation_engine.recommend_products(
            crop_name="Cotton",
            disease_name="Aphids / Thrips damage",
            is_healthy=False
        )
        self.assertIsInstance(res, ProductRecommendationResult)
        self.assertFalse(res.is_healthy)
        self.assertGreater(len(res.recommendations), 0, "Should return verified NACL insecticide recommendations")
        
        categories = [r.category for r in res.recommendations]
        self.assertIn("Insecticides", categories)


if __name__ == "__main__":
    unittest.main()
