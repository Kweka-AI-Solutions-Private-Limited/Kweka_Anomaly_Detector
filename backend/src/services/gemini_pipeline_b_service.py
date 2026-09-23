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
from datetime import datetime
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
from services.vlm_service import get_gemini_api_key

GEMINI_PIPELINE_B_PROMPT = """You are an industrial visual quality-inspection system.

The supplied image contains one or more physical instances of the same type of product.

Inspect the ENTIRE image and identify EVERY visible product instance.

For each product instance:

1. Locate the COMPLETE physical product.
2. Return its bounding box in ORIGINAL IMAGE coordinates (normalized 0 to 1000 scale).
3. Determine whether it is:
   - GOOD
   - DEFECTIVE
4. Base the decision ONLY on visible physical evidence.
5. If DEFECTIVE:
   - identify the visible defect type
   - locate the defective region
   - return a tight defect bounding box (normalized 0 to 1000 scale)
   - describe the visible defect
   - assign severity: LOW, MEDIUM, or HIGH
6. If GOOD:
   - defect_type = null
   - defect_bbox = null
   - defect_description = null
   - severity = null
7. Return confidence from 0.0 to 1.0.

Inspect each product independently.

Do NOT assume all products have the same status.

Differences in position, scale, orientation, lighting, background, reflections, or camera perspective are NOT defects by themselves.

Look carefully for subtle physical defects including scratches, surface damage, missing/manipulated components, thread damage, deformation, cracks, dents, and other visible manufacturing abnormalities.

Do not invent defects that are not visually supported.

IMPORTANT:
- Product bounding boxes must refer to the ORIGINAL image in [ymin, xmin, ymax, xmax] 0-1000 scale.
- Defect bounding boxes must refer to the ORIGINAL image in [ymin, xmin, ymax, xmax] 0-1000 scale.
- Coordinates must be [ymin, xmin, ymax, xmax].
- Find EVERY visible product.
- Do not omit products.

Return ONLY valid JSON matching this schema:

{
  "instances": [
    {
      "product_id": 1,
      "product_bbox": [ymin, xmin, ymax, xmax],
      "status": "GOOD",
      "defect_type": null,
      "defect_bbox": null,
      "defect_description": null,
      "severity": null,
      "confidence": 0.95
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
    Draws green bounding boxes for GOOD products, red for DEFECTIVE products,
    and highlighted yellow bounding boxes for defect regions.
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
        p_bbox = inst.get("bbox")
        is_defective = inst.get("status") == "DEFECTIVE"
        conf = inst.get("confidence", 0.0)

        if p_bbox:
            px, py, pw, ph = p_bbox["x"], p_bbox["y"], p_bbox["width"], p_bbox["height"]
            box_color = (0, 0, 235) if is_defective else (0, 200, 80) # BGR Red or Green

            # Thick box border
            cv2.rectangle(vis_bgr, (px, py), (px + pw, py + ph), box_color, 2)

            # Label box
            status_txt = "DEFECTIVE" if is_defective else "GOOD"
            label = f"#{inst_id} {status_txt} ({int(conf * 100)}%)"

            txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            label_bg_y1 = max(0, py - txt_size[1] - 6)
            cv2.rectangle(vis_bgr, (px, label_bg_y1), (px + txt_size[0] + 8, py), box_color, -1)
            cv2.putText(vis_bgr, label, (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw defect bbox if present
        def_bbox = inst.get("defect_bbox")
        if def_bbox and is_defective:
            dx, dy, dw, dh = def_bbox["x"], def_bbox["y"], def_bbox["width"], def_bbox["height"]

            # Draw yellow highlighted defect region fill
            defect_overlay = vis_bgr.copy()
            cv2.rectangle(defect_overlay, (dx, dy), (dx + dw, dy + dh), (0, 255, 255), -1)
            cv2.addWeighted(defect_overlay, 0.4, vis_bgr, 0.6, 0, vis_bgr)

            # Draw yellow outline
            cv2.rectangle(vis_bgr, (dx, dy), (dx + dw, dy + dh), (0, 255, 255), 2)

            # Defect text label
            dtype = inst.get("defect_type") or "DEFECT"
            dlabel = f"DEFECT: {dtype}"
            cv2.putText(vis_bgr, dlabel, (dx, max(15, dy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(output_path), vis_bgr)


def run_gemini_only_multi_instance_inspection(
    db: Database,
    model_id: str,
    upload_file: UploadFile,
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
        created_at=datetime.utcnow()
    )

    data = insp_doc.model_dump() if hasattr(insp_doc, "model_dump") else insp_doc.dict()
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
                temperature=0.1
            )
        )

        gemini_raw_text = response.text or "{}"
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
        instance_results: List[InstanceResultSchema] = []

        for idx, item in enumerate(raw_instances, start=1):
            product_id = item.get("product_id") or idx
            p_box_raw = item.get("product_bbox")
            status_raw = str(item.get("status") or "GOOD").upper()
            is_defective = status_raw == "DEFECTIVE"

            p_pixel = norm_to_pixel_bbox(p_box_raw, orig_w, orig_h)
            if not p_pixel:
                continue

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

            parsed_instances.append({
                "instance_id": product_id,
                "status": status_raw,
                "bbox": p_pixel,
                "defect_bbox": d_pixel,
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
                explanation=item.get("defect_description") or ("Defective product instance identified." if is_defective else "No physical defects identified."),
                severity=item.get("severity") if is_defective else "NONE",
                location=f"x:{d_pixel['x']}, y:{d_pixel['y']}" if d_pixel else "N/A"
            )

            # Check if PatchCore model artifacts are available to compute real crop anomaly scores & heatmaps
            patchcore_score = 0.0
            # Pure-Gemini Pipeline B status is determined directly by Gemini VLM.
            # Anomaly score is set to None because Gemini produces qualitative visual classification rather than PatchCore feature distances.
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
            instance_results.append(inst_obj)

        # 4. Generate Deterministic Composite Inspection Visualization
        composite_vis_path = target_path.parent / "composite_heatmap.png"
        generate_gemini_inspection_visualization(img_bgr, parsed_instances, composite_vis_path)

        try:
            rel_comp_uri = str(composite_vis_path.relative_to(storage_root)).replace("\\", "/")
        except ValueError:
            rel_comp_uri = f"storage/inspections/{inspection_id}/composite_heatmap.png"

        # 5. Aggregate Verdict
        pass_cnt = sum(1 for i in instance_results if i.prediction and i.prediction.status == "PASS")
        reject_cnt = sum(1 for i in instance_results if i.prediction and i.prediction.status == "REJECT")

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
            created_at=datetime.utcnow()
        )

        res_data = res_doc.model_dump() if hasattr(res_doc, "model_dump") else res_doc.dict()
        db.ad_inspections.insert_one(res_data)

        db.ad_inspections.update_one(
            {"_id": ObjectId(inspection_id)},
            {"$set": {
                "status": "completed",
                "processing_time_ms": total_time_ms,
                "completed_at": datetime.utcnow()
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
