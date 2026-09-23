"""
dashboard_service.py — Dashboard Aggregation Service
------------------------------------------------------
Aggregates persisted inspections, runs, models, VLM analysis, and feedback data
for the Point 7B Dashboard. Performs database-side query filtering and structured
KPI computation.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from bson import ObjectId


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None


def get_dashboard_summary(
    db: Any,
    model_id: Optional[str] = None,
    model_version_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Returns structured dashboard metrics aggregated across inspections, runs, models, and feedback.
    Supports scope filters: model_id, model_version_id, start_date, end_date.
    """
    # 1. Build Query for Inspections
    query: Dict[str, Any] = {}

    if model_id and model_id != "all":
        try:
            m_obj_id = ObjectId(model_id)
            query["$or"] = [{"model_id": m_obj_id}, {"model_id": model_id}]
        except Exception:
            query["model_id"] = model_id

    if model_version_id and model_version_id != "all":
        try:
            v_obj_id = ObjectId(model_version_id)
            if "$or" in query:
                query["model_version_id"] = {"$in": [v_obj_id, model_version_id]}
            else:
                query["$or"] = [{"model_version_id": v_obj_id}, {"model_version_id": model_version_id}]
        except Exception:
            query["model_version_id"] = model_version_id

    dt_start = parse_date(start_date)
    dt_end = parse_date(end_date)
    if dt_start or dt_end:
        date_q: Dict[str, Any] = {}
        if dt_start:
            date_q["$gte"] = dt_start
        if dt_end:
            date_q["$lte"] = dt_end
        query["created_at"] = date_q

    # 2. Fetch Matching Inspections
    raw_inspections = list(db.ad_inspections.find(query).sort("created_at", -1))

    # Build Metadata Maps
    models_list = list(db.ad_models.find())
    model_map = {}
    for m in models_list:
        m_id = str(m.get("_id") or m.get("id"))
        model_map[m_id] = m.get("name", "Unknown Model")

    versions_list = list(db.ad_models.find())
    version_map = {}
    for v in versions_list:
        v_id = str(v.get("_id") or v.get("id"))
        version_map[v_id] = v.get("version_number", 1)

    runs_list = list(db.ad_inspection_runs.find())
    run_map = {}
    for r in runs_list:
        r_id = str(r.get("_id") or r.get("id"))
        run_map[r_id] = r.get("run_number")

    # Fetch Inspection Results
    inspection_ids = [insp.get("_id") for insp in raw_inspections if insp.get("_id")]
    results_list = list(db.ad_inspections.find({"inspection_id": {"$in": inspection_ids}}))
    results_map = {}
    for res in results_list:
        insp_ref = str(res.get("inspection_id"))
        results_map[insp_ref] = res

    # 3. KPI Aggregations
    pass_count = 0
    reject_count = 0
    error_count = 0

    daily_trend: Dict[str, Dict[str, Any]] = {}
    defect_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    model_counts: Dict[str, int] = {}
    recent_inspections_data = []

    # Hotspot Matrix Initialization (5x5 grid)
    grid_matrix = [[0 for _ in range(5)] for _ in range(5)]
    inspections_with_bbox = 0

    for insp in raw_inspections:
        insp_id_str = str(insp.get("_id") or insp.get("id"))
        res = results_map.get(insp_id_str) or insp

        pred = res.get("prediction") or insp.get("prediction") or {}
        vlm = res.get("vlm_analysis") or insp.get("vlm_analysis") or {}

        raw_status = (pred.get("status") or insp.get("status") or "normal").lower()
        if raw_status in ["normal", "pass", "completed"]:
            # Check prediction status specifically if insp.status is completed
            p_stat = (pred.get("status") or "normal").lower()
            if p_stat in ["anomalous", "reject"]:
                verdict = "REJECT"
                reject_count += 1
            elif p_stat in ["error"]:
                verdict = "ERROR"
                error_count += 1
            else:
                verdict = "PASS"
                pass_count += 1
        elif raw_status in ["anomalous", "reject"]:
            verdict = "REJECT"
            reject_count += 1
        elif raw_status in ["error", "failed"]:
            verdict = "ERROR"
            error_count += 1
        else:
            verdict = "PASS"
            pass_count += 1

        # Model Distribution
        m_id_val = str(insp.get("model_id"))
        m_name = (insp.get("model_name") or model_map.get(m_id_val) or "Inspection Model")
        model_counts[m_name] = model_counts.get(m_name, 0) + 1

        # Daily Trend
        c_at = insp.get("created_at")
        if isinstance(c_at, datetime):
            day_str = c_at.strftime("%Y-%m-%d")
        else:
            day_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if day_str not in daily_trend:
            daily_trend[day_str] = {"date": day_str, "total": 0, "pass": 0, "reject": 0}
        daily_trend[day_str]["total"] += 1
        if verdict == "PASS":
            daily_trend[day_str]["pass"] += 1
        elif verdict == "REJECT":
            daily_trend[day_str]["reject"] += 1

        # Defect & Severity Distributions & Hotspot Grid (Non-PASS)
        if verdict == "REJECT":
            d_type = vlm.get("defect_type")
            if d_type and d_type not in ["Not required", "None", "N/A", "skipped"]:
                defect_counts[d_type] = defect_counts.get(d_type, 0) + 1

            sev = vlm.get("severity") or pred.get("severity")
            if sev and sev.capitalize() in severity_counts:
                severity_counts[sev.capitalize()] += 1

            # Bbox hotspot grid overlap calculation (PatchCore localization.bbox)
            loc = res.get("localization") or insp.get("localization") or {}
            raw_bbox = (
                loc.get("bbox") if isinstance(loc, dict)
                else (getattr(loc, "bbox", None) if hasattr(loc, "bbox") else None)
            )

            bbox_coords = None
            if raw_bbox:
                if isinstance(raw_bbox, dict):
                    bx, by, bw, bh = raw_bbox.get("x"), raw_bbox.get("y"), raw_bbox.get("width"), raw_bbox.get("height")
                elif hasattr(raw_bbox, "x"):
                    bx, by, bw, bh = getattr(raw_bbox, "x", None), getattr(raw_bbox, "y", None), getattr(raw_bbox, "width", None), getattr(raw_bbox, "height", None)
                else:
                    bx = by = bw = bh = None

                if bx is not None and by is not None and bw is not None and bh is not None:
                    try:
                        fbx, fby, fbw, fbh = float(bx), float(by), float(bw), float(bh)
                        if fbw > 0 and fbh > 0:
                            bbox_coords = (fbx, fby, fbw, fbh)
                    except (ValueError, TypeError):
                        bbox_coords = None

            if bbox_coords:
                inspections_with_bbox += 1
                bx1, by1, bw, bh = bbox_coords
                bx2 = bx1 + bw
                by2 = by1 + bh

                W = max(256.0, bx2)
                H = max(256.0, by2)
                cell_w = W / 5.0
                cell_h = H / 5.0

                for r in range(5):
                    cy1 = r * cell_h
                    cy2 = (r + 1) * cell_h
                    for c in range(5):
                        cx1 = c * cell_w
                        cx2 = (c + 1) * cell_w

                        ox1 = max(cx1, bx1)
                        ox2 = min(cx2, bx2)
                        oy1 = max(cy1, by1)
                        oy2 = min(cy2, by2)

                        ow = max(0.0, ox2 - ox1)
                        oh = max(0.0, oy2 - oy1)

                        if (ow * oh) > 0:
                            grid_matrix[r][c] += 1

        # Build Recent Inspection Item
        if len(recent_inspections_data) < 10:
            v_id_val = str(insp.get("model_version_id"))
            ver_num = version_map.get(v_id_val)
            r_id_val = str(insp.get("run_id")) if insp.get("run_id") else None
            r_num = insp.get("run_number") or (run_map.get(r_id_val) if r_id_val else None)

            input_data = insp.get("input") or {}
            filename = insp.get("filename") or input_data.get("filename") or "sample.png"
            storage_uri = insp.get("storage_uri") or input_data.get("storage_uri")

            recent_inspections_data.append({
                "id": insp_id_str,
                "filename": filename,
                "storage_uri": storage_uri,
                "model_name": m_name,
                "model_version_number": ver_num,
                "run_number": r_num,
                "run_id": r_id_val,
                "verdict": verdict,
                "anomaly_score": pred.get("anomaly_score"),
                "defect_type": vlm.get("defect_type") if verdict == "REJECT" else "N/A",
                "severity": (vlm.get("severity") or pred.get("severity")) if verdict == "REJECT" else "N/A",
                "created_at": c_at.isoformat() if isinstance(c_at, datetime) else str(c_at)
            })

    # Total Inspections = PASS + REJECT + ERROR (Requirement 2)
    total_inspections = pass_count + reject_count + error_count

    # Pass Rate = PASS / (PASS + REJECT) * 100 (Requirement 3 - Errors excluded from denominator)
    valid_denom = pass_count + reject_count
    pass_rate = round((pass_count / valid_denom) * 100, 1) if valid_denom > 0 else 0.0

    # 4. Runs Summary (Filtered by Model / Version / Date scope)
    run_query: Dict[str, Any] = {}
    if model_id and model_id != "all":
        try:
            m_obj_id = ObjectId(model_id)
            run_query["$or"] = [{"model_id": m_obj_id}, {"model_id": model_id}]
        except Exception:
            run_query["model_id"] = model_id

    if model_version_id and model_version_id != "all":
        try:
            v_obj_id = ObjectId(model_version_id)
            if "$or" in run_query:
                run_query["model_version_id"] = {"$in": [v_obj_id, model_version_id]}
            else:
                run_query["$or"] = [{"model_version_id": v_obj_id}, {"model_version_id": model_version_id}]
        except Exception:
            run_query["model_version_id"] = model_version_id

    matching_runs = list(db.ad_inspection_runs.find(run_query).sort("created_at", -1))
    total_runs = len(matching_runs)
    completed_runs = sum(1 for r in matching_runs if r.get("status") == "completed")
    failed_runs = sum(1 for r in matching_runs if r.get("status") in ["failed", "error"])

    recent_runs_summary = []
    for r in matching_runs[:5]:
        r_m_id = str(r.get("model_id"))
        r_v_id = str(r.get("model_version_id"))
        recent_runs_summary.append({
            "id": str(r.get("_id") or r.get("id")),
            "run_number": r.get("run_number"),
            "model_name": model_map.get(r_m_id, "Model"),
            "version_number": version_map.get(r_v_id, 1),
            "total_images": r.get("total_images", 0),
            "status": r.get("status", "completed")
        })

    # Active Models Count
    active_models = len(set(str(insp.get("model_id")) for insp in raw_inspections if insp.get("model_id")))
    if active_models == 0:
        active_models_count = len([m for m in models_list if m.get("status") == "active"])
        active_models = active_models_count if active_models_count > 0 else len(models_list)

    # 5. Feedback Summary (Scoped to Model & Date via matching inspection IDs - Requirement 4!)
    matched_insp_obj_ids = [insp.get("_id") for insp in raw_inspections if insp.get("_id")]
    matched_insp_str_ids = [str(i_id) for i_id in matched_insp_obj_ids]
    all_matched_ids = matched_insp_obj_ids + matched_insp_str_ids

    feedback_query = {"inspection_id": {"$in": all_matched_ids}} if all_matched_ids else {"inspection_id": "__none__"}
    scoped_feedback = list(db.ad_feedback.find(feedback_query))

    fb_total = len(scoped_feedback)
    fb_categories = {
        "correct": 0,
        "false_positive": 0,
        "false_negative": 0,
        "wrong_defect_type": 0,
        "wrong_location": 0,
        "wrong_severity": 0
    }

    for fb in scoped_feedback:
        det_fb = fb.get("detection_feedback")
        if det_fb == "correct":
            fb_categories["correct"] += 1
        elif det_fb == "false_positive":
            fb_categories["false_positive"] += 1
        elif det_fb == "false_negative":
            fb_categories["false_negative"] += 1

        vlm_cats = fb.get("vlm_feedback_categories") or []
        for cat in vlm_cats:
            if cat in fb_categories:
                fb_categories[cat] += 1

    # Format Distributions & Hotspot Grid Summary
    sorted_trend = [daily_trend[k] for k in sorted(daily_trend.keys())]
    sorted_defects = [{"defect_type": k, "count": v} for k, v in sorted(defect_counts.items(), key=lambda x: x[1], reverse=True)]
    sorted_models = [{"model_name": k, "count": v} for k, v in sorted(model_counts.items(), key=lambda x: x[1], reverse=True)]

    max_cell_count = max(max(r) for r in grid_matrix) if grid_matrix else 0
    hotspot_cells = []
    for r in range(5):
        for c in range(5):
            cnt = grid_matrix[r][c]
            intensity = round(cnt / max_cell_count, 2) if max_cell_count > 0 else 0.0
            hotspot_cells.append({
                "row": r,
                "col": c,
                "count": cnt,
                "intensity": intensity
            })

    hotspot_analysis = {
        "grid_size": 5,
        "total_anomalous_inspections": reject_count,
        "inspections_with_bbox": inspections_with_bbox,
        "matrix": grid_matrix,
        "max_cell_count": max_cell_count,
        "cells": hotspot_cells
    }

    return {
        "kpi": {
            "total_inspections": total_inspections,
            "pass_count": pass_count,
            "reject_count": reject_count,
            "error_count": error_count,
            "pass_rate": pass_rate,
            "total_runs": total_runs,
            "active_models": active_models
        },
        "pass_vs_reject": {
            "pass_count": pass_count,
            "reject_count": reject_count,
            "pass_percentage": round((pass_count / valid_denom * 100), 1) if valid_denom > 0 else 0.0,
            "reject_percentage": round((reject_count / valid_denom * 100), 1) if valid_denom > 0 else 0.0
        },
        "inspection_trend": sorted_trend,
        "defect_distribution": sorted_defects,
        "anomaly_hotspot_analysis": hotspot_analysis,
        "severity_distribution": severity_counts,
        "model_distribution": sorted_models,
        "run_summary": {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "recent_runs": recent_runs_summary
        },
        "feedback_summary": {
            "total_feedback": fb_total,
            "categories": fb_categories
        },
        "recent_inspections": recent_inspections_data
    }

