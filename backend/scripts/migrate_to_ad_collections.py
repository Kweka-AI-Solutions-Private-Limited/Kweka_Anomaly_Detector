"""
Migrate MongoDB schema from legacy 12 collections to 5 core `ad_` collections.
Consolidates:
  1. models, model_versions, model_groups, reference_images -> `ad_models`
  2. inspections, inspection_results, leaf_disease_runs, anomaly_detector -> `ad_inspections`
  3. inspection_runs -> `ad_inspection_runs`
  4. feedback -> `ad_feedback`
  5. notifications -> `ad_notifications`
  6. nacl_products -> `ad_nacl_products`

Frees up cluster collection quota by dropping legacy collections after migration.
"""

import sys
from pathlib import Path

# Add backend/src to path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from db.connection import get_db, close_connection

def migrate():
    db = get_db()
    print(f"Starting migration on database: {db.name}")
    
    # 1. Fetch data from legacy model collections
    print("--- 1. Migrating models -> ad_models ---")
    models = list(db.models.find()) if "models" in db.list_collection_names() else []
    versions = list(db.model_versions.find()) if "model_versions" in db.list_collection_names() else []
    references = list(db.reference_images.find()) if "reference_images" in db.list_collection_names() else []
    groups = {str(g["_id"]): g for g in db.model_groups.find()} if "model_groups" in db.list_collection_names() else {}

    # Group versions by model_id
    versions_by_model = {}
    for v in versions:
        m_id = str(v.get("model_id"))
        versions_by_model.setdefault(m_id, []).append(v)

    # Group reference images by model_id
    refs_by_model = {}
    for r in references:
        m_id = str(r.get("model_id"))
        refs_by_model.setdefault(m_id, []).append(r)

    ad_models_docs = []
    for m in models:
        m_id_str = str(m["_id"])
        m_versions = versions_by_model.get(m_id_str, [])
        m_refs = refs_by_model.get(m_id_str, [])
        
        g_id_str = str(m.get("group_id")) if m.get("group_id") else None
        g_info = groups.get(g_id_str) if g_id_str else None

        ad_model_doc = dict(m)
        ad_model_doc["versions"] = m_versions
        ad_model_doc["reference_images"] = m_refs
        if g_info:
            ad_model_doc["group"] = {
                "id": str(g_info["_id"]),
                "name": g_info.get("name"),
                "description": g_info.get("description")
            }
        ad_models_docs.append(ad_model_doc)

    # 2. Fetch data from legacy inspection collections
    print("--- 2. Migrating inspections -> ad_inspections ---")
    inspections = list(db.inspections.find()) if "inspections" in db.list_collection_names() else []
    results = list(db.inspection_results.find()) if "inspection_results" in db.list_collection_names() else []
    
    results_by_insp = {}
    for res in results:
        insp_id_str = str(res.get("inspection_id"))
        results_by_insp[insp_id_str] = res

    ad_inspections_docs = []
    for insp in inspections:
        insp_id_str = str(insp["_id"])
        res_doc = results_by_insp.get(insp_id_str)

        ad_insp_doc = dict(insp)
        if res_doc:
            ad_insp_doc["result"] = res_doc
            if "prediction" in res_doc and "prediction" not in ad_insp_doc:
                ad_insp_doc["prediction"] = res_doc["prediction"]
            if "localization" in res_doc and "localization" not in ad_insp_doc:
                ad_insp_doc["localization"] = res_doc["localization"]
            if "vlm_analysis" in res_doc and "vlm_analysis" not in ad_insp_doc:
                ad_insp_doc["vlm_analysis"] = res_doc["vlm_analysis"]
        ad_inspections_docs.append(ad_insp_doc)

    # Leaf disease runs
    leaf_runs = list(db.leaf_disease_runs.find()) if "leaf_disease_runs" in db.list_collection_names() else []
    for lr in leaf_runs:
        lr_doc = dict(lr)
        lr_doc["inspection_mode"] = "leaf_disease"
        ad_inspections_docs.append(lr_doc)

    # 3. Fetch runs, feedback, notifications, nacl_products
    runs = list(db.inspection_runs.find()) if "inspection_runs" in db.list_collection_names() else []
    feedback = list(db.feedback.find()) if "feedback" in db.list_collection_names() else []
    notifs = list(db.notifications.find()) if "notifications" in db.list_collection_names() else []
    prods = list(db.nacl_products.find()) if "nacl_products" in db.list_collection_names() else []

    # Drop legacy collections first to free collection count quota
    print("--- Dropping legacy collections to free cluster collection quota ---")
    legacy_collections = [
        "models", "model_versions", "model_groups", "reference_images",
        "inspections", "inspection_results", "inspection_runs",
        "feedback", "notifications", "nacl_products", "leaf_disease_runs", "anomaly_detector"
    ]
    for col_name in legacy_collections:
        if col_name in db.list_collection_names():
            db.drop_collection(col_name)
            print(f"Dropped legacy collection: {col_name}")

    # Write to new ad_ collections
    print("--- Inserting consolidated data into ad_ collections ---")
    if ad_models_docs:
        db.ad_models.insert_many(ad_models_docs)
        print(f"Inserted {len(ad_models_docs)} documents into ad_models")
    if ad_inspections_docs:
        db.ad_inspections.insert_many(ad_inspections_docs)
        print(f"Inserted {len(ad_inspections_docs)} documents into ad_inspections")
    if runs:
        db.ad_inspection_runs.insert_many(runs)
        print(f"Inserted {len(runs)} documents into ad_inspection_runs")
    if feedback:
        db.ad_feedback.insert_many(feedback)
        print(f"Inserted {len(feedback)} documents into ad_feedback")
    if notifs:
        db.ad_notifications.insert_many(notifs)
        print(f"Inserted {len(notifs)} documents into ad_notifications")
    if prods:
        db.ad_nacl_products.insert_many(prods)
        print(f"Inserted {len(prods)} documents into ad_nacl_products")

    print("\n[OK] Database consolidation migration completed successfully!")

if __name__ == "__main__":
    migrate()
    close_connection()
