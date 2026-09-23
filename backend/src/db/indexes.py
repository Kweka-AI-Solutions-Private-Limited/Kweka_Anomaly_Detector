"""
InspectAI MongoDB Programmatic Index Creation
----------------------------------------------
Idempotently creates indexes and unique constraints across the core `ad_` collections:
  1. ad_models
  2. ad_inspections
  3. ad_inspection_runs
  4. ad_feedback
  5. ad_notifications
  6. ad_nacl_products
"""

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database


def ensure_indexes(db: Database) -> dict:
    """
    Programmatically creates all required single & compound indexes,
    including unique constraints across core ad_ collections. Safe to run repeatedly (idempotent).
    Returns a summary dictionary of created indexes per collection.
    """
    results = {}

    # 1. AD_MODELS
    idx_models = db.ad_models.create_index([("status", ASCENDING)], background=True)
    idx_models_upd = db.ad_models.create_index([("updated_at", DESCENDING)], background=True)
    idx_models_grp = db.ad_models.create_index([("group_id", ASCENDING)], background=True)
    results["ad_models"] = [idx_models, idx_models_upd, idx_models_grp]

    # 2. AD_INSPECTIONS
    idx_insp_m = db.ad_inspections.create_index([("model_id", ASCENDING)], background=True)
    idx_insp_v = db.ad_inspections.create_index([("model_version_id", ASCENDING)], background=True)
    idx_insp_r = db.ad_inspections.create_index([("run_id", ASCENDING)], background=True)
    idx_insp_s = db.ad_inspections.create_index([("status", ASCENDING)], background=True)
    idx_insp_c = db.ad_inspections.create_index([("created_at", DESCENDING)], background=True)
    idx_insp_mc = db.ad_inspections.create_index([("model_id", ASCENDING), ("created_at", DESCENDING)], background=True)
    results["ad_inspections"] = [idx_insp_m, idx_insp_v, idx_insp_r, idx_insp_s, idx_insp_c, idx_insp_mc]

    # 3. AD_INSPECTION_RUNS
    idx_ir_uniq = db.ad_inspection_runs.create_index(
        [("model_id", ASCENDING), ("run_number", ASCENDING)],
        unique=True,
        background=True
    )
    idx_ir_mc = db.ad_inspection_runs.create_index([("model_id", ASCENDING), ("created_at", DESCENDING)], background=True)
    idx_ir_s = db.ad_inspection_runs.create_index([("status", ASCENDING)], background=True)
    results["ad_inspection_runs"] = [idx_ir_uniq, idx_ir_mc, idx_ir_s]

    # 4. AD_FEEDBACK
    idx_fb_i = db.ad_feedback.create_index([("inspection_id", ASCENDING)], background=True)
    idx_fb_m = db.ad_feedback.create_index([("model_id", ASCENDING)], background=True)
    idx_fb_v = db.ad_feedback.create_index([("model_version_id", ASCENDING)], background=True)
    idx_fb_c = db.ad_feedback.create_index([("created_at", DESCENDING)], background=True)
    results["ad_feedback"] = [idx_fb_i, idx_fb_m, idx_fb_v, idx_fb_c]

    # 5. AD_NOTIFICATIONS
    idx_notif_uniq = db.ad_notifications.create_index([("idempotency_key", ASCENDING)], unique=True, background=True)
    idx_notif_read = db.ad_notifications.create_index([("read", ASCENDING), ("created_at", DESCENDING)], background=True)
    idx_notif_created = db.ad_notifications.create_index([("created_at", DESCENDING)], background=True)
    results["ad_notifications"] = [idx_notif_uniq, idx_notif_read, idx_notif_created]

    return results


