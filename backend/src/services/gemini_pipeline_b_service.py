"""
InspectAI Pure-Gemini Pipeline B Service
----------------------------------------
Standalone Gemini-Only Multi-Product Inspection Engine.
Inspects multi-product composite images directly via a single structured Gemini VLM call,
bypassing PatchCore, crop isolation preprocessing, and GOOD reference image requirements.
"""

import os
import time
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from bson import ObjectId
from fastapi import HTTPException, UploadFile
from pymongo.database import Database

from google import genai
from google.genai import types

from db.schemas import (
    InspectionSchema, InspectionInput, InspectionResultSchema,
    PredictionOutput, LocalizationOutput, BoundingBox,
    VLMAnalysisSchema, OverallPredictionOutput, InstanceResultSchema
)
from services.storage_service import save_inspection_image, get_storage_base_dir
from services.model_service import get_model, get_model_version
from services.freeform_contour_service import extract_product_contour, extract_defect_contour
from services.vlm_service import get_gemini_api_key

GEMINI_PIPELINE_B_PROMPT = """You are an industrial visual quality-inspection vision system.

Your primary duty is to perform an EXHAUSTIVE physical product inventory and strict quality assessment for multi-product images.

Follow this strict 2-step protocol:

STEP 1: SYSTEMATIC SPATIAL PRODUCT INVENTORY & EXHAUSTIVE ENUMERATION
- Perform a rigorous 2D spatial grid sweep from TOP-TO-BOTTOM and LEFT-TO-RIGHT across the ENTIRE surface.
- Look specifically for vertical screws (pointing up or down), horizontal screws, diagonal screws, perimeter/edge screws, and screws sandwiched between other adjacent items.
- PERIMETER & ORIENTATION CHECK: Pay special attention to vertical screws positioned along the right/left perimeters or sandwiched beside diagonal screws (e.g. top-right vertical screw pointing downwards).
- COUNT VERIFICATION: In multi-product composite grids, count all physical items first. Every visible screw MUST receive exactly one instance entry with a valid `product_bbox`.
- Assign sequential `product_id` numbers starting at 1 ordered spatially (top-to-bottom, left-to-right).

STEP 2: INDEPENDENT QUALITY INSPECTION
Inspect each enumerated product instance independently across all 5 physical regions:
1. HEAD: Inspect for cracks, chips, deformation, malformed geometry, abnormal edges, or head pattern distortion.
2. SHANK: Inspect for bending, abnormal curvature, body axis deformation, shank swelling/gaps, or local material distortion.
3. THREADS: Inspect for malformed thread ridges, thread pitch irregularity, abnormal material distortion along thread crests/roots, missing thread sections, damaged thread peaks, flattened threads, bent threads, or local thread deformation.
4. TIP: Inspect for bent tip, curved tip, broken tip, blunted/flattened tip, or malformed tip taper geometry.
5. SURFACE: Inspect for scratches, gouges, cracks, abnormal material loss, or abnormal material protrusions.

STRICT DEFECT DETECTION RULES:
- Inspect EVERY product thoroughly from head to tip.
- If ANY physical defect is presentâ€”including a bent/curved tip, bent shank, deformed/flattened thread ridges, head crack, or localized material distortionâ€”you MUST classify the product as "DEFECTIVE".
- Do NOT classify a clean, intact product as defective based solely on uniform metallic sheen or normal lighting reflections.
- However, do NOT ignore genuine physical defects such as bent tips, bent shanks, or malformed threads. Any visible physical deviation from clean screw geometry MUST be flagged as "DEFECTIVE".

OUTPUT REQUIREMENTS:
- For GOOD instances:
   * status = "GOOD"
   * defect_type = null, defect_description = null, defect_bbox = null, severity = null
   * inspection_evidence = concise physical confirmation of intact geometry
- For DEFECTIVE instances:
   * status = "DEFECTIVE"
   * defect_type = short string (e.g. "bent_tip", "bent_shank", "damaged_threads", "head_crack", "surface_gouge")
   * defect_description = concise visual evidence of physical damage
   * defect_bbox = tight bounding box `[ymin, xmin, ymax, xmax]` in 0-1000 scale enclosing ONLY the defective region
   * severity = "LOW", "MEDIUM", or "HIGH"
- Assign `confidence`: float between 0.0 and 1.0.

Return ONLY valid JSON matching this exact schema:

{
  "instances": [
    {
      "product_id": 1,
      "product_bbox": [ymin, xmin, ymax, xmax],
      "status": "GOOD",
      "defect_type": null,
      "defect_description": null,
      "defect_bbox": null,
      "severity": null,
      "confidence": 0.98,
      "inspection_evidence": "Head, shank, threads, and tip geometry intact"
    },
    {
      "product_id": 2,
      "product_bbox": [ymin, xmin, ymax, xmax],
      "status": "DEFECTIVE",
      "defect_type": "bent_tip",
      "defect_description": "Screw tip exhibits visible bending and taper deformation",
      "defect_bbox": [ymin, xmin, ymax, xmax],
      "severity": "MEDIUM",
      "confidence": 0.95,
      "inspection_evidence": "Visible curvature at screw tip"
    }
  ]
}"""


def norm_to_pixel_bbox(box_1000: Any, img_w: int, img_h: int) -> Optional[Dict[str, int]]:
    """Converts 0-1000 normalized coordinates [ymin, xmin, ymax, xmax] or [x1, y1, x2, y2] into pixel {x, y, width, height}."""
    if not box_1000 or not isinstance(box_1000, (list, tuple)) or len(box_1000) != 4:
        return None
    try:
        val1, val2, val3, val4 = [float(v) for v in box_1000]

        # Determine scale (0-1 float, 0-1000 integer, or direct pixel)
        max_val = max(val1, val2, val3, val4)
        if max_val <= 1.05:
            scale = 1.0
        elif max_val <= 1005:
            scale = 1000.0
        else:
            scale = None

        if scale is not None:
            # Assume ymin, xmin, ymax, xmax if standard Gemini format, or x1, y1, x2, y2
            # Check orientation: if val1 > val3, swap
            if val1 > val3:
                val1, val3 = val3, val1
            if val2 > val4:
                val2, val4 = val4, val2

            ymin, xmin, ymax, xmax = val1, val2, val3, val4

            pixel_xmin = max(0, min(img_w - 1, int(round((xmin / scale) * img_w))))
            pixel_ymin = max(0, min(img_h - 1, int(round((ymin / scale) * img_h))))
            pixel_xmax = max(pixel_xmin + 1, min(img_w, int(round((xmax / scale) * img_w))))
            pixel_ymax = max(pixel_ymin + 1, min(img_h, int(round((ymax / scale) * img_h))))
        else:
            # Direct pixel coordinates
            pixel_xmin = max(0, min(img_w - 1, int(round(val1))))
            pixel_ymin = max(0, min(img_h - 1, int(round(val2))))
            pixel_xmax = max(pixel_xmin + 1, min(img_w, int(round(val3))))
            pixel_ymax = max(pixel_ymin + 1, min(img_h, int(round(val4))))

        w = max(1, pixel_xmax - pixel_xmin)
        h = max(1, pixel_ymax - pixel_ymin)
        return {"x": pixel_xmin, "y": pixel_ymin, "width": w, "height": h}
    except Exception:
        return None


def generate_gemini_inspection_visualization(
    img_bgr: np.ndarray,
    instance_results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Generates a deterministic GEMINI INSPECTION VISUALIZATION image on the composite surface.
    Draws organic freeform polylines for product outlines and filled yellow polygons for defect regions.
    Includes an explicit header label 'GEMINI INSPECTION VISUALIZATION'.
    """
    vis_bgr = img_bgr.copy()
    h, w = vis_bgr.shape[:2]

    # Draw semi-transparent header bar
    header_h = max(40, int(h * 0.05))
    overlay = vis_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, vis_bgr, 0.15, 0, vis_bgr)

    cv2.putText(
        vis_bgr,
        "GEMINI INSPECTION VISUALIZATION / DIRECT VLM ANALYSIS",
        (15, int(header_h * 0.65)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, w / 1800.0),
        (56, 189, 248),  # Sky blue font
        2,
        cv2.LINE_AA
    )

    for inst in instance_results:
        inst_id = inst.get("instance_id")
        is_defective = inst.get("status") == "DEFECTIVE"
        conf = inst.get("confidence", 0.0)
        prod_poly = inst.get("product_polygon") or []
        def_poly = inst.get("defect_polygon") or []
        p_bbox = inst.get("bbox")

        # Convert normalized 0-1000 product polygon points to pixel coordinates [[x, y], ...]
        poly_pts = []
        if prod_poly and len(prod_poly) >= 3:
            for pt in prod_poly:
                py_norm, px_norm = pt[0], pt[1]
                px_pix = max(0, min(w - 1, int(round((px_norm / 1000.0) * w))))
                py_pix = max(0, min(h - 1, int(round((py_norm / 1000.0) * h))))
                poly_pts.append([px_pix, py_pix])
        elif p_bbox:
            px, py, pw, ph = p_bbox["x"], p_bbox["y"], p_bbox["width"], p_bbox["height"]
            poly_pts = [[px, py], [px + pw, py], [px + pw, py + ph], [px, py + ph]]

        if poly_pts:
            box_color = (0, 0, 235) if is_defective else (0, 200, 80) # BGR Red or Green
            pts_arr = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))

            # Draw organic polygon contour outline
            cv2.polylines(vis_bgr, [pts_arr], isClosed=True, color=box_color, thickness=2, lineType=cv2.LINE_AA)

            # Label badge positioned at top vertex
            top_pt = min(poly_pts, key=lambda p: (p[1], p[0]))
            lbl_x, lbl_y = top_pt[0], top_pt[1]

            status_txt = "DEFECTIVE" if is_defective else "GOOD"
            label = f"#{inst_id} {status_txt} ({int(conf * 100)}%)"

            txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            label_bg_y1 = max(0, lbl_y - txt_size[1] - 6)
            cv2.rectangle(vis_bgr, (lbl_x, label_bg_y1), (lbl_x + txt_size[0] + 8, lbl_y), box_color, -1)
            cv2.putText(vis_bgr, label, (lbl_x + 4, lbl_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw organic defect highlight polygon if present
        if def_poly and len(def_poly) >= 3 and is_defective:
            dpoly_pts = []
            for pt in def_poly:
                py_norm, px_norm = pt[0], pt[1]
                px_pix = max(0, min(w - 1, int(round((px_norm / 1000.0) * w))))
                py_pix = max(0, min(h - 1, int(round((py_norm / 1000.0) * h))))
                dpoly_pts.append([px_pix, py_pix])

            dpts_arr = np.array(dpoly_pts, dtype=np.int32).reshape((-1, 1, 2))

            # Semi-transparent yellow fill
            defect_overlay = vis_bgr.copy()
            cv2.fillPoly(defect_overlay, [dpts_arr], (0, 255, 255))
            cv2.addWeighted(defect_overlay, 0.4, vis_bgr, 0.6, 0, vis_bgr)

            # Yellow outline
            cv2.polylines(vis_bgr, [dpts_arr], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            dtype = inst.get("defect_type") or "DEFECT"
            dlabel = f"DEFECT: {dtype}"
            top_dpt = min(dpoly_pts, key=lambda p: p[1])
            cv2.putText(vis_bgr, dlabel, (top_dpt[0], max(15, top_dpt[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(output_path), vis_bgr)


def run_gemini_only_multi_instance_inspection(
    db: Database,
    model_id: str,
    upload_file: UploadFile,
    user_id: Optional[str] = "usr_default",
    threshold_override: Optional[float] = None,
    min_instance_area: int = 500,
    max_instances: int = 20,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes PURE GEMINI PIPELINE B Multi-Instance Inspection.
    Sends the multi-product image to Gemini Vision in a single call.
    Parses instances, saves individual crops, creates deterministic inspection visualization,
    and returns standardized result schema compatible with the frontend.
    """
    start_total_time = time.time()

    model = get_model(db, model_id)
    model_status = (model.get("status") or "").lower()
    if model_status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model.get('name', model_id)}' is currently in '{model_status.upper()}' state and cannot be used for inspections."
        )

    active_version_id = str(model.get("active_version_id") or "")
    model_threshold = 27.0
    if active_version_id:
        try:
            ver = get_model_version(db, model_id, active_version_id)
            if ver:
                model_threshold = float(ver.get("calibration", {}).get("threshold", 27.0))
        except Exception:
            pass

    filename_str = upload_file.filename or "pure_gemini_multi_instance.png"

    # 1. Create preliminary inspection document
    insp_doc = InspectionSchema(
        model_id=ObjectId(model_id),
        model_version_id=ObjectId(active_version_id) if active_version_id and ObjectId.is_valid(active_version_id) else None,
        run_id=ObjectId(run_id) if run_id and ObjectId.is_valid(run_id) else None,
        inspection_mode="multi_instance",
        status="processing",
        input=InspectionInput(
            storage_uri="",
            filename=filename_str,
            width=256,
            height=256
        ),
        created_at=datetime.now(timezone.utc)
    )

    data = insp_doc.model_dump()
    data["user_id"] = user_id or "usr_default"
    res = db.ad_inspections.insert_one(data)
    inspection_id = str(res.inserted_id)

    try:
        # 2. Save original uploaded multi-product image
        target_path, relative_uri, file_size, checksum = save_inspection_image(inspection_id, upload_file)
        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {"input.storage_uri": relative_uri}}
        )

        img_bgr = cv2.imread(str(target_path))
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to decode uploaded image.")
        orig_h, orig_w = img_bgr.shape[:2]

        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {"input.width": orig_w, "input.height": orig_h}}
        )

        # 3. Call Gemini VLM (Single Call)
        api_key = get_gemini_api_key()
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not configured.")

        gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        client = genai.Client(api_key=api_key)

        with open(target_path, "rb") as f:
            image_bytes = f.read()

        mime_type = "image/png"
        if filename_str.lower().endswith((".jpg", ".jpeg")):
            mime_type = "image/jpeg"

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            GEMINI_PIPELINE_B_PROMPT
        ]

        response = client.models.generate_content(
            model=gemini_model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=1024) if hasattr(types, "ThinkingConfig") else None
            )
        )

        gemini_raw_text = response.text or ""
        if not gemini_raw_text and hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            parts = getattr(getattr(cand, "content", None), "parts", None)
            if parts:
                for p in parts:
                    txt = getattr(p, "text", "")
                    if txt and not getattr(p, "thought", False):
                        gemini_raw_text = txt
                        break

        if not gemini_raw_text:
            gemini_raw_text = "{}"

        try:
            parsed = json.loads(gemini_raw_text)
        except Exception:
            # Attempt json block extraction if needed
            cleaned = gemini_raw_text.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            parsed = json.loads(cleaned)

        raw_instances = parsed.get("instances", [])
        if not isinstance(raw_instances, list):
            raw_instances = []

        storage_root = get_storage_base_dir()
        crops_dir = target_path.parent / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        parsed_instances: List[Dict[str, Any]] = []
        instance_results: List[Dict[str, Any]] = []

        for idx, item in enumerate(raw_instances, start=1):
            product_id = item.get("product_id") or idx
            p_box_raw = item.get("product_bbox")
            status_raw = str(item.get("status") or "GOOD").upper()
            is_defective = status_raw == "DEFECTIVE"

            p_pixel = norm_to_pixel_bbox(p_box_raw, orig_w, orig_h)
            if not p_pixel:
                # Construct safe fallback bounding box covering full image frame if product_bbox is malformed/missing
                p_pixel = {"x": 0, "y": 0, "width": orig_w, "height": orig_h}
                confidence = 0.50
            else:
                confidence = float(item.get("confidence") or 0.90)

            # Extract defect box if present
            d_box_raw = item.get("defect_bbox")
            d_pixel = norm_to_pixel_bbox(d_box_raw, orig_w, orig_h) if is_defective and d_box_raw else None

            # Save instance crop PNG for frontend drawer
            cx, cy, cw, ch = p_pixel["x"], p_pixel["y"], p_pixel["width"], p_pixel["height"]
            crop_bgr = img_bgr[cy : cy + ch, cx : cx + cw]

            crop_filename = f"instance_{product_id}.png"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop_bgr)

            try:
                crop_uri = str(crop_path.relative_to(storage_root)).replace("\\", "/")
            except ValueError:
                crop_uri = f"storage/inspections/{inspection_id}/crops/{crop_filename}"

            confidence = float(item.get("confidence") or 0.90)

            # Extract freeform organic visual contours (Visualization-Only, 0 impact on classification/Gemini)
            prod_poly = extract_product_contour(img_bgr, p_pixel) if p_pixel else []
            def_poly = extract_defect_contour(img_bgr, d_pixel) if is_defective and d_pixel else []

            parsed_instances.append({
                "instance_id": product_id,
                "status": status_raw,
                "bbox": p_pixel,
                "defect_bbox": d_pixel,
                "product_polygon": prod_poly,
                "defect_polygon": def_poly,
                "defect_type": item.get("defect_type"),
                "confidence": confidence
            })

            orig_bbox = BoundingBox(x=cx, y=cy, width=cw, height=ch)

            mapped_defect_bbox = None
            if d_pixel:
                mapped_defect_bbox = BoundingBox(
                    x=d_pixel["x"],
                    y=d_pixel["y"],
                    width=d_pixel["width"],
                    height=d_pixel["height"]
                )

            vlm_analysis = VLMAnalysisSchema(
                status="completed",
                provider="gemini",
                defect_type=item.get("defect_type") if is_defective else "Normal",
                explanation=item.get("defect_description") or item.get("inspection_evidence") or ("Defective product instance identified." if is_defective else "Head, shank, threads, and tip geometry intact."),
                severity=item.get("severity") if is_defective else "NONE",
                location=f"x:{d_pixel['x']}, y:{d_pixel['y']}" if d_pixel else "N/A"
            )

            # Check if PatchCore model artifacts are available to compute real crop anomaly scores & heatmaps
            patchcore_score = 0.0
            inst_status = "REJECT" if is_defective else "PASS"

            inst_obj = InstanceResultSchema(
                instance_id=product_id,
                bbox=orig_bbox,
                padded_bbox=orig_bbox,
                crop_storage_uri=crop_uri,
                detection_confidence=confidence,
                status="completed",
                prediction=PredictionOutput(
                    status=inst_status,
                    anomaly_score=None,
                    threshold=None,
                    severity=item.get("severity") if is_defective else "NONE"
                ),
                localization=LocalizationOutput(
                    bbox=mapped_defect_bbox,
                    heatmap_uri=None
                ),
                vlm_analysis=vlm_analysis
            )
            # Attach freeform visual contours to instance dict for API output
            inst_dict = inst_obj.model_dump()
            inst_dict["product_polygon"] = prod_poly
            inst_dict["defect_polygon"] = def_poly
            instance_results.append(inst_dict)

        # 4. Generate Deterministic Composite Inspection Visualization
        composite_vis_path = target_path.parent / "composite_heatmap.png"
        generate_gemini_inspection_visualization(img_bgr, parsed_instances, composite_vis_path)

        try:
            rel_comp_uri = str(composite_vis_path.relative_to(storage_root)).replace("\\", "/")
        except ValueError:
            rel_comp_uri = f"storage/inspections/{inspection_id}/composite_heatmap.png"

        # 5. Aggregate Verdict
        pass_cnt = sum(1 for i in instance_results if i.get("prediction", {}).get("status") == "PASS")
        reject_cnt = sum(1 for i in instance_results if i.get("prediction", {}).get("status") == "REJECT")

        overall_status = "REJECT" if reject_cnt > 0 else "PASS"
        overall_msg = f"{reject_cnt} of {len(instance_results)} instance(s) failed Gemini inspection." if reject_cnt > 0 else "All instances passed Gemini inspection."

        overall_pred = OverallPredictionOutput(
            status=overall_status,
            total_instances=len(instance_results),
            pass_count=pass_cnt,
            reject_count=reject_cnt,
            error_count=0,
            max_anomaly_score=0.0,
            threshold=model_threshold,
            reason=None,
            message=overall_msg
        )

        total_time_ms = round((time.time() - start_total_time) * 1000, 1)

        res_doc = InspectionResultSchema(
            inspection_id=ObjectId(inspection_id),
            inspection_mode="multi_instance",
            overall_prediction=overall_pred,
            instances=instance_results,
            composite_heatmap_uri=rel_comp_uri,
            processing_stats={
                "detection_time_ms": total_time_ms,
                "patchcore_time_ms": 0.0,
                "total_time_ms": total_time_ms,
                "instance_count": float(len(instance_results)),
                "image_width": orig_w,
                "image_height": orig_h,
                "engine": "gemini_only",
                "model": gemini_model_name
            },
            created_at=datetime.now(timezone.utc)
        )

        res_data = res_doc.model_dump()
        db.ad_inspections.insert_one(res_data)

        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {
                "status": "completed",
                "processing_time_ms": total_time_ms,
                "completed_at": datetime.now(timezone.utc)
            }}
        )

        from services.inspection_service import normalize_inspection_payload
        doc = db.ad_inspections.find_one({"_id": ObjectId(inspection_id)})
        if not doc:
            raise HTTPException(status_code=500, detail="Failed to retrieve completed inspection document.")
        return normalize_inspection_payload(doc, res_data)

    except Exception as e:
        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {"status": "failed"}}
        )
        raise HTTPException(status_code=500, detail=f"Pure Gemini Pipeline B inspection failed: {str(e)}")


