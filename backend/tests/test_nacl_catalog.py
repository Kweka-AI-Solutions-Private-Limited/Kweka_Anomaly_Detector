"""
test_nacl_catalog.py — Unit Tests for NACL Catalog Schema, Scraper & Database Service (Phase 2)
-----------------------------------------------------------------------------------------------
"""

import unittest
from pathlib import Path
from schemas.nacl_catalog import NACLProduct, CropDosageRule
from services.leaf_disease.nacl_scraper import NACLScraper
from db.nacl_product_db import seed_nacl_products, query_nacl_products


class TestNACLCatalog(unittest.TestCase):
    """Test suite covering NACL catalog schema parsing, scraper logic, and MongoDB service."""

    def test_nacl_product_schema_validation(self):
        """Verify NACLProduct schema instantiation and serialization."""
        prod = NACLProduct(
            product_id="test-product",
            product_name="Test Product",
            category="Fungicides",
            product_url="https://naclind.com/products/fungicides/test/",
            active_ingredient="Difenoconazole 25% EC",
            chemical_class="Triazole",
            mode_of_action="Systemic action inhibiting ergosterol biosynthesis.",
            pack_sizes=["100ml", "250ml", "500ml"],
            key_benefits=["Effective control against Pear Rust", "Rapid absorption"],
            crop_applications=[
                CropDosageRule(crop="Pear", target_pest_or_disease="European Pear Rust", dosage="100 ml/acre")
            ]
        )
        self.assertEqual(prod.product_id, "test-product")
        self.assertEqual(prod.category, "Fungicides")
        self.assertEqual(len(prod.crop_applications), 1)
        self.assertEqual(prod.crop_applications[0].crop, "Pear")

    def test_crop_dosage_line_parser(self):
        """Verify regex parsing of crop dosage text lines in scraper."""
        scraper = NACLScraper()
        rule = scraper._parse_crop_dosage_line("Chilli: 320–400 ml/acre")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.crop, "Chilli")
        self.assertEqual(rule.dosage, "320–400 ml/acre")

        invalid_rule = scraper._parse_crop_dosage_line("Random non-dosage text with no colon")
        self.assertIsNone(invalid_rule)

    def test_catalog_json_file_exists_and_populated(self):
        """Verify nacl_catalog.json file exists and contains scraped products."""
        catalog_path = Path(__file__).resolve().parent.parent / "src" / "data" / "nacl_catalog.json"
        self.assertTrue(catalog_path.exists(), "nacl_catalog.json should exist")

        import json
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertGreater(len(data), 0, "Catalog should contain scraped products")
        first = data[0]
        self.assertIn("product_id", first)
        self.assertIn("product_name", first)
        self.assertIn("category", first)

    def test_mongodb_product_query(self):
        """Verify MongoDB NACL product queries."""
        # Query for Grape products
        results = query_nacl_products(crop_name="Grape", category="Insecticides")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Should return at least 1 product for Grape/Insecticides")


if __name__ == "__main__":
    unittest.main()
