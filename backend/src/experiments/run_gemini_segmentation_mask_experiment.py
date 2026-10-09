import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv

# Ensure backend/src is in sys.path
backend_src = Path(__file__).resolve().parent.parent
if str(backend_src) not in sys.path:
    sys.path.insert(0, str(backend_src))

load_dotenv(backend_src.parent / ".env")

from google import genai
from google.genai import types

def run_tight_segmentation_experiment():
    print("==================================================")
    print("TIGHT GEMINI DEFECT SEGMENTATION MASK EXPERIMENT")
    print("==================================================")

    output_dir = backend_src.parent / "storage" / "experiments" / "gemini_tight_segmentation_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    composite_dir = backend_src.parent / "storage" / "experiments" / "gemini_reference_comparison_diagnostic" / "composites"
    composite_files = sorted(list(composite_dir.glob("composite_*.png")))
    if not composite_files:
        print(f"[ERROR] No composite images found in {composite_dir}")
        return

    print(f"[FOUND {len(composite_files)} COMPOSITE IMAGES]")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable is missing.")
        return

    gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    prompt = """
You are a high-precision industrial defect inspection vision system.
Analyze this multi-product image containing multiple physical product instances (screws/fasteners).

Your task:
1. Locate EVERY visible physical product instance (`product_bbox`).
2. Inspect each screw carefully for defects (surface scratch, bent tip, bent shank, deformed thread, head crack, discoloration).
3. Classify each screw as "GOOD" or "DEFECTIVE".
4. For DEFECTIVE instances, pinpoint the EXACT visual evidence of the defect and provide BOTH a tight defect bounding box AND a tight defect segmentation mask polygon.

CRITICAL LOCALIZATION RULES FOR DEFECT MASK POLYGON:
- The `defect_mask_polygon` MUST cover ONLY the exact physical region that constitutes the defect.
- DO NOT return:
  - the entire product
  - the entire screw body or shank
  - a large contextual area around the defect
  - the product_bbox as the defect mask
  - a generic rectangular region where the defect happens to be located
- SPECIFIC EXAMPLES:
  * SCRATCH: Polygon must tightly trace ONLY the thin visible scratch line/mark itself.
  * BENT SHANK / BENT TIP: Polygon must cover ONLY the specific bent apex or deformed segment where geometry deviates from straight, NOT the entire screw shank.
  * DEFORMED THREAD: Polygon must cover ONLY the specific malformed thread ridge(s), NOT all threads.
  * DAMAGED / CRACKED HEAD: Polygon must cover ONLY the crack or damaged notch, NOT the entire screw head.
- The `defect_bbox` MUST be the tight 2D bounding rectangle `[ymin, xmin, ymax, xmax]` enclosing ONLY the `defect_mask_polygon`.
- UNCERTAINTY RULE: If there is insufficient visual evidence to localize the exact defect boundary tightly, set `"defect_mask_polygon": null` and `"defect_bbox": null`. Do NOT invent a large approximate polygon.

Format requirements:
Return ONLY a JSON object with key "instances".

Each item in "instances" must contain:
- "product_id": integer (1, 2, 3...)
- "status": "GOOD" or "DEFECTIVE"
- "confidence": float between 0.0 and 1.0 (inspection confidence)
- "product_bbox": [ymin, xmin, ymax, xmax] normalized 0-1000 for the entire product
- "defect_type": string describing defect (e.g. "surface_scratch", "bent_tip", "deformed_thread", "head_crack") or null if GOOD
- "defect_description": short text describing exact visual evidence found, or null if GOOD
- "defect_bbox": [ymin, xmin, ymax, xmax] normalized 0-1000 tightly bounding ONLY the defect region, or null if GOOD
- "defect_mask_polygon": list of [y, x] vertices (normalized 0-1000) tightly tracing ONLY the exact defect contour, or null if GOOD.
- "severity": "NONE", "LOW", "MEDIUM", "HIGH", or "CRITICAL"

Important:
- Return ONLY valid JSON matching this format.
- Normalized 0-1000 coordinates mean 0=top/left and 1000=bottom/right.
"""

    results_summary = []

    for comp_path in composite_files:
        print(f"\n==================================================")
        print(f"TESTING COMPOSITE: {comp_path.name}")
        print(f"==================================================")
        img_bgr = cv2.imread(str(comp_path))
        if img_bgr is None:
            continue
        img_h, img_w = img_bgr.shape[:2]

        with open(comp_path, "rb") as f:
            image_bytes = f.read()

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt
        ]

        start_time = time.time()
        try:
            response = client.models.generate_content(
                model=gemini_model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            latency_ms = round((time.time() - start_time) * 1000, 2)

            usage_meta = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage_meta, "prompt_token_count", None) if usage_meta else None
            candidates_tokens = getattr(usage_meta, "candidates_token_count", None) if usage_meta else None
            total_tokens = getattr(usage_meta, "total_token_count", None) if usage_meta else None

            raw_text = response.text or "{}"
            
            # Save Raw JSON Response
            raw_response_path = output_dir / f"raw_response_tight_{comp_path.stem}.json"
            with open(raw_response_path, "w", encoding="utf-8") as f:
                f.write(raw_text)

            try:
                parsed = json.loads(raw_text)
            except Exception:
                cleaned = raw_text.strip()
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                elif "```" in cleaned:
                    cleaned = cleaned.split("```")[1].split("```")[0].strip()
                parsed = json.loads(cleaned)

            instances = parsed.get("instances", [])
            print(f"  [RESULT] Detected {len(instances)} instances | Latency: {latency_ms}ms | Tokens: {total_tokens}")

            vis_img = img_bgr.copy()
            vis_img_copy = img_bgr.copy()

            def norm_to_pixel_box(bbox_norm):
                if not bbox_norm or len(bbox_norm) != 4:
                    return None
                ymin, xmin, ymax, xmax = bbox_norm
                x1 = int(round((xmin / 1000.0) * img_w))
                y1 = int(round((ymin / 1000.0) * img_h))
                x2 = int(round((xmax / 1000.0) * img_w))
                y2 = int(round((ymax / 1000.0) * img_h))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
                return (x1, y1, x2, y2)

            def norm_polygon_to_pixels(poly_norm):
                if not poly_norm or not isinstance(poly_norm, list):
                    return None
                pts = []
                for pt in poly_norm:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        y_n, x_n = pt[0], pt[1]
                        px = int(round((float(x_n) / 1000.0) * img_w))
                        py = int(round((float(y_n) / 1000.0) * img_h))
                        pts.append([px, py])
                if len(pts) < 3:
                    return None
                return np.array(pts, dtype=np.int32)

            mask_count = 0
            defective_count = 0

            for inst in instances:
                pid = inst.get("product_id")
                status = str(inst.get("status", "UNKNOWN")).upper()
                confidence = inst.get("confidence")
                p_bbox_norm = inst.get("product_bbox")
                d_type = inst.get("defect_type")
                d_desc = inst.get("defect_description")
                d_bbox_norm = inst.get("defect_bbox")
                d_mask_norm = inst.get("defect_mask_polygon")
                severity = inst.get("severity")

                if status == "DEFECTIVE":
                    defective_count += 1
                    print(f"    - Instance #{pid}: DEFECTIVE | Type: {d_type} | Severity: {severity}")
                    print(f"      Evidence: {d_desc}")
                    print(f"      Defect BBox: {d_bbox_norm}")
                    print(f"      Defect Mask: {d_mask_norm}")

                p_box = norm_to_pixel_box(p_bbox_norm)
                if p_box:
                    px1, py1, px2, py2 = p_box
                    color = (255, 220, 0) if status == "GOOD" else (0, 140, 255)
                    cv2.rectangle(vis_img, (px1, py1), (px2, py2), color, 2)
                    cv2.putText(vis_img, f"#{pid} {status}", (px1, max(15, py1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                d_box = norm_to_pixel_box(d_bbox_norm)
                if d_box:
                    dx1, dy1, dx2, dy2 = d_box
                    cv2.rectangle(vis_img, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                    cv2.putText(vis_img, f"DEFECT ({d_type})", (dx1, max(15, dy1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                poly_pts = norm_polygon_to_pixels(d_mask_norm)
                if poly_pts is not None:
                    mask_count += 1
                    cv2.fillPoly(vis_img_copy, [poly_pts], (0, 255, 255))
                    cv2.polylines(vis_img, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=2)

            alpha = 0.45
            cv2.addWeighted(vis_img_copy, alpha, vis_img, 1 - alpha, 0, vis_img)

            # Legend
            cv2.rectangle(vis_img, (10, 10), (440, 110), (20, 20, 20), -1)
            cv2.putText(vis_img, "TIGHT GEMINI DEFECT MASK EXPERIMENT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(vis_img, "Cyan/Orange Box = Product BBox", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1)
            cv2.putText(vis_img, "Red Box = Defect BBox", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            cv2.putText(vis_img, "Yellow Polygon = Tight Gemini Defect Mask", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

            vis_output_path = output_dir / f"visualization_tight_{comp_path.stem}.png"
            cv2.imwrite(str(vis_output_path), vis_img)
            print(f"  [SAVED VISUALIZATION] {vis_output_path} (Defective: {defective_count}, Masks: {mask_count})")

            results_summary.append({
                "composite": comp_path.name,
                "instances": len(instances),
                "defective": defective_count,
                "masks": mask_count,
                "latency_ms": latency_ms,
                "tokens": total_tokens
            })

        except Exception as e:
            print(f"  [ERROR on {comp_path.name}] {e}")

    summary_file = output_dir / "summary_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[SAVED SUMMARY RESULTS] {summary_file}")

if __name__ == "__main__":
    run_tight_segmentation_experiment()
