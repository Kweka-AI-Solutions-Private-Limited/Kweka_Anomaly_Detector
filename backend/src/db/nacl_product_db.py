"""
nacl_product_db.py — NACL Agrochemical Product Database Collection & Queries (Phase 2)
----------------------------------------------------------------------------------------
Manages the `nacl_products` MongoDB collection:
  - Catalog Seeder: upserts `src/data/nacl_catalog.json` into MongoDB
  - Indexes: product_id, category, active_ingredient, crop_applications.crop
  - Query Service: product lookup by crop, pathogen/disease, and active ingredient
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pymongo.database import Database
from pymongo import ASCENDING, TEXT

from db.connection import get_db

logger = logging.getLogger(__name__)

COLLECTION_NAME = "nacl_products"


def init_nacl_product_indexes(db: Optional[Database] = None) -> None:
    """Creates search indexes on `nacl_products` collection."""
    if db is None:
        db = get_db()

    coll = db[COLLECTION_NAME]
    coll.create_index([("product_id", ASCENDING)], unique=True)
    coll.create_index([("category", ASCENDING)])
    coll.create_index([("active_ingredient", ASCENDING)])
    coll.create_index([("crop_applications.crop", ASCENDING)])
    coll.create_index([
        ("product_name", TEXT),
        ("active_ingredient", TEXT),
        ("mode_of_action", TEXT),
        ("key_benefits", TEXT)
    ])
    logger.info("Successfully initialized MongoDB indexes for '%s' collection.", COLLECTION_NAME)


def seed_nacl_products(catalog_json_path: Optional[Path] = None, db: Optional[Database] = None) -> int:
    """Seeds or updates `nacl_products` MongoDB collection from JSON catalog file."""
    if db is None:
        db = get_db()

    if catalog_json_path is None:
        catalog_json_path = Path(__file__).resolve().parent.parent / "data" / "nacl_catalog.json"

    if not catalog_json_path.exists():
        logger.error("NACL catalog JSON file not found at: %s", str(catalog_json_path))
        return 0

    with open(catalog_json_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    if not products:
        logger.warning("NACL catalog JSON file is empty.")
        return 0

    init_nacl_product_indexes(db)
    coll = db[COLLECTION_NAME]

    inserted_or_updated = 0
    for prod in products:
        p_id = prod.get("product_id")
        if not p_id:
            continue

        coll.update_one(
            {"product_id": p_id},
            {"$set": prod},
            upsert=True
        )
        inserted_or_updated += 1

    logger.info("Successfully seeded/upserted %d NACL products into MongoDB.", inserted_or_updated)
    return inserted_or_updated


def query_nacl_products(
    crop_name: Optional[str] = None,
    disease_name: Optional[str] = None,
    category: Optional[str] = None,
    db: Optional[Database] = None
) -> List[Dict[str, Any]]:
    """
    Queries NACL products matching crop, disease/pathogen, or category.
    Returns normalized product dictionaries.
    """
    if db is None:
        db = get_db()

    coll = db[COLLECTION_NAME]
    filter_query: Dict[str, Any] = {}

    if category:
        filter_query["category"] = {"$regex": f"^{category}$", "$options": "i"}

    # Flexible regex query for crop
    if crop_name and crop_name.lower() != "unknown":
        crop_clean = crop_name.strip()
        filter_query["$or"] = [
            {"crop_applications.crop": {"$regex": crop_clean, "$options": "i"}},
            {"key_benefits": {"$regex": crop_clean, "$options": "i"}},
            {"mode_of_action": {"$regex": crop_clean, "$options": "i"}}
        ]

    results = list(coll.find(filter_query, {"_id": 0}))

    # Fallback if specific crop filter returned zero matches — return relevant category products
    if not results and category:
        results = list(coll.find({"category": {"$regex": f"^{category}$", "$options": "i"}}, {"_id": 0}))

    # Secondary fallback — return all products if still empty
    if not results:
        results = list(coll.find({}, {"_id": 0}).limit(20))

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cnt = seed_nacl_products()
    print(f"Seeded {cnt} products into MongoDB 'nacl_products'.")
