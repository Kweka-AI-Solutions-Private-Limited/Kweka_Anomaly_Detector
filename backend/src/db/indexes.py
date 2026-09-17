"""
InspectAI MongoDB Programmatic Index Creation
----------------------------------------------
Idempotently creates indexes and unique constraints across the 6 core collections:
  1. models
  2. model_versions
  3. reference_images
  4. inspections
  5. inspection_results
  6. feedback
"""

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database


def ensure_indexes(db: Database) -> dict:
    """
    Programmatically creates all required single & compound indexes,
    including unique constraints. Safe to run repeatedly (idempotent).
    Returns a summary dictionary of created indexes per collection.
    """
    results = {}

    # 0. MODEL_GROUPS
    idx_mg_name = db.model_groups.create_index([("name", ASCENDING)], unique=True, background=True)
    results["model_groups"] = [idx_mg_name]

    # 1. MODELS
    idx_models = db.models.create_index([("status", ASCENDING)], background=True)
    idx_models_upd = db.models.create_index([("updated_at", DESCENDING)], background=True)
    idx_models_grp = db.models.create_index([("group_id", ASCENDING)], background=True)
    idx_models_grp_stat = db.models.create_index([("group_id", ASCENDING), ("status", ASCENDING)], background=True)
    results["models"] = [idx_models, idx_models_upd, idx_models_grp, idx_models_grp_stat]

    # 2. MODEL_VERSIONS
    idx_mv_mid = db.model_versions.create_index([("model_id", ASCENDING)], background=True)
    # Unique constraint on (model_id, version_number)
    idx_mv_uniq = db.model_versions.create_index(
        [("model_id", ASCENDING), ("version_number", ASCENDING)],
        unique=True,
        background=True
    )
    idx_mv_stat = db.model_versions.create_index([("model_id", ASCENDING), ("status", ASCENDING)], background=True)
    results["model_versions"] = [idx_mv_mid, idx_mv_uniq, idx_mv_stat]

    # 3. REFERENCE_IMAGES
    idx_ref_m = db.reference_images.create_index([("model_id", ASCENDING)], background=True)
    idx_ref_v = db.reference_images.create_index([("version_id", ASCENDING)], background=True)
    idx_ref_mv = db.reference_images.create_index([("model_id", ASCENDING), ("version_id", ASCENDING)], background=True)
    results["reference_images"] = [idx_ref_m, idx_ref_v, idx_ref_mv]

    # 4. INSPECTIONS
    idx_insp_m = db.inspections.create_index([("model_id", ASCENDING)], background=True)
    idx_insp_v = db.inspections.create_index([("model_version_id", ASCENDING)], background=True)
    idx_insp_r = db.inspections.create_index([("run_id", ASCENDING)], background=True)
    idx_insp_s = db.inspections.create_index([("status", ASCENDING)], background=True)
    idx_insp_c = db.inspections.create_index([("created_at", DESCENDING)], background=True)
    idx_insp_mc = db.inspections.create_index([("model_id", ASCENDING), ("created_at", DESCENDING)], background=True)
    results["inspections"] = [idx_insp_m, idx_insp_v, idx_insp_r, idx_insp_s, idx_insp_c, idx_insp_mc]

    # 5. INSPECTION_RUNS
    # Unique constraint on (model_id, run_number)
    idx_ir_uniq = db.inspection_runs.create_index(
        [("model_id", ASCENDING), ("run_number", ASCENDING)],
        unique=True,
        background=True
    )
    idx_ir_mc = db.inspection_runs.create_index([("model_id", ASCENDING), ("created_at", DESCENDING)], background=True)
    idx_ir_s = db.inspection_runs.create_index([("status", ASCENDING)], background=True)
    results["inspection_runs"] = [idx_ir_uniq, idx_ir_mc, idx_ir_s]

    # 6. INSPECTION_RESULTS
    # Unique constraint on inspection_id
    idx_res_uniq = db.inspection_results.create_index([("inspection_id", ASCENDING)], unique=True, background=True)
    results["inspection_results"] = [idx_res_uniq]

    # 7. FEEDBACK
    idx_fb_i = db.feedback.create_index([("inspection_id", ASCENDING)], background=True)
    idx_fb_m = db.feedback.create_index([("model_id", ASCENDING)], background=True)
    idx_fb_v = db.feedback.create_index([("model_version_id", ASCENDING)], background=True)
    idx_fb_c = db.feedback.create_index([("created_at", DESCENDING)], background=True)
    results["feedback"] = [idx_fb_i, idx_fb_m, idx_fb_v, idx_fb_c]

    # 8. NOTIFICATIONS
    idx_notif_uniq = db.notifications.create_index([("idempotency_key", ASCENDING)], unique=True, background=True)
    idx_notif_read = db.notifications.create_index([("read", ASCENDING), ("created_at", DESCENDING)], background=True)
    idx_notif_created = db.notifications.create_index([("created_at", DESCENDING)], background=True)
    results["notifications"] = [idx_notif_uniq, idx_notif_read, idx_notif_created]

    return results

