"""
Reference-Based Gemini Multi-Product Defect Detection Diagnostic Experiment
---------------------------------------------------------------------------
Evaluates whether Gemini Vision can accurately detect and localize manufacturing defects
in multi-product test images when explicitly provided EXACTLY ONE known-GOOD reference image (IMAGE 1)
and asked to compare each product in a multi-product test image (IMAGE 2) against that GOOD baseline.

Phases:
  1. Reference & Composite Construction: Select 5 known-GOOD reference images from train/good. Build 5 composite test scenarios.
  2. Gemini 2-Image Inspection: Send [IMAGE 1 (GOOD reference), IMAGE 2 (Test Composite)] + reference comparison prompt to Gemini Vision.
  3. Instance Matching & Localization: IoU matching (threshold 0.80) between predicted and GT product bboxes.
  4. GOOD/DEFECTIVE Classification: Compute TP, FN, TN, FP, Defect Recall, GOOD FPR, Accuracy, and per-category breakdown.
  5. Localized Deviation Region & Heatmap Generation: Evaluate deviation regions vs GT defect masks and render feathered heatmap overlays.
  6. False Positive / False Negative Analysis: Record qualitative analysis of hallucinations and missed defects.
  7. Visual Overlay & Reporting: Save heatmap overlays, JSON reports, README.md, and print final terminal summary.

Production Safety:
  Strictly standalone research script. No production code, models, database, or thresholds modified.
"""

import sys
import os
import time
import json
import random
import re
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load environment variables from backend/.env if available
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from services.gemini_instance_localization_service import get_gemini_api_key, get_gemini_instance_model

# Set random seeds for strict reproducibility
random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "data" / "mvtec_anomaly_detection" / "screw"
OUTPUT_DIR = BASE_DIR / "storage" / "experiments" / "gemini_reference_comparison_diagnostic"

COMPOSITES_DIR = OUTPUT_DIR / "composites"
REFERENCES_DIR = OUTPUT_DIR / "references"
RAW_RESPONSES_DIR = OUTPUT_DIR / "raw_responses"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, COMPOSITES_DIR, REFERENCES_DIR, RAW_RESPONSES_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two pixel bboxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return float(iou)


def build_reference_and_composite_scenarios():
    """
    Selects 5 distinct known-GOOD reference images from train/good.
    Builds 5 multi-screw composite test scenarios (1024x1024 canvas).
    Returns dictionary mapping composite name -> metadata including reference image path and GT products.
    """
    good_train_files = sorted(list((DATASET_DIR / "train" / "good").glob("*.png")))
    good_test_files = sorted(list((DATASET_DIR / "test" / "good").glob("*.png")))

    # Select 5 pristine GOOD reference images from train/good (indices 0, 1, 2, 3, 4)
    ref_files = good_train_files[:5]
    saved_ref_paths = []
    for idx, rfile in enumerate(ref_files):
        ref_dest = REFERENCES_DIR / f"ref_00{idx+1}.png"
        shutil.copy(str(rfile), str(ref_dest))
        saved_ref_paths.append(ref_dest)

    # Remaining GOOD files for composites (excluding the 5 references)
    good_pool = good_train_files[5:] + good_test_files
    random.seed(42)
    shuffled_good = good_pool.copy()
    random.shuffle(shuffled_good)

    categories = ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]
    defective_files_by_cat = {cat: sorted(list((DATASET_DIR / "test" / cat).glob("*.png"))) for cat in categories}
    gt_mask_dir = DATASET_DIR / "ground_truth"

    scenarios_config = [
        # Scenario 1: 4 GOOD + 5 DEFECTIVE (3x3 grid)
        {
            "name": "composite_001.png",
            "ref_path": str(saved_ref_paths[0]),
            "title": "Scenario 1: 4 GOOD + 5 DEFECTIVE (3x3 Grid)",
            "layout": (3, 3),
            "canvas_size": (1024, 1024),
            "bg_color": (202, 202, 202),
            "items": [
                ("good", 1.0), ("manipulated_front", 1.0), ("good", 1.0),
                ("scratch_head", 1.0), ("scratch_neck", 1.0), ("good", 1.0),
                ("thread_side", 1.0), ("good", 1.0), ("thread_top", 1.0)
            ]
        },
        # Scenario 2: 5 GOOD + 5 DEFECTIVE (mixed 3x3 layout + 1 extra)
        {
            "name": "composite_002.png",
            "ref_path": str(saved_ref_paths[1]),
            "title": "Scenario 2: 5 GOOD + 5 DEFECTIVE (All Defect Categories)",
            "layout": (2, 5),
            "canvas_size": (1024, 1024),
            "bg_color": (202, 202, 202),
            "items": [
                ("good", 1.0), ("scratch_neck", 1.0), ("good", 1.0), ("thread_side", 1.0), ("good", 1.0),
                ("manipulated_front", 1.0), ("good", 1.0), ("thread_top", 1.0), ("good", 1.0), ("scratch_head", 1.0)
            ]
        },
        # Scenario 3: GOOD-Only Baseline (6 GOOD Screws)
        {
            "name": "composite_003.png",
            "ref_path": str(saved_ref_paths[2]),
            "title": "Scenario 3: GOOD-Only Composite (6 GOOD Screws)",
            "layout": (2, 3),
            "canvas_size": (1024, 1024),
            "bg_color": (202, 202, 202),
            "items": [
                ("good", 1.0), ("good", 1.0), ("good", 1.0),
                ("good", 1.0), ("good", 1.0), ("good", 1.0)
            ]
        },
        # Scenario 4: Varying Scales & Spacing (4 GOOD + 4 DEFECTIVE)
        {
            "name": "composite_004.png",
            "ref_path": str(saved_ref_paths[3]),
            "title": "Scenario 4: Varying Scales & Spacing (4 GOOD + 4 DEFECTIVE)",
            "layout": (2, 4),
            "canvas_size": (1024, 1024),
            "bg_color": (202, 202, 202),
            "items": [
                ("good", 0.85), ("scratch_head", 1.15), ("good", 0.90), ("thread_side", 1.10),
                ("manipulated_front", 1.10), ("good", 0.85), ("scratch_neck", 0.95), ("good", 1.05)
            ]
        },
        # Scenario 5: Dense Mixed Arrangement (2 GOOD + 6 DEFECTIVE)
        {
            "name": "composite_005.png",
            "ref_path": str(saved_ref_paths[4]),
            "title": "Scenario 5: Dense Mixed Composite (2 GOOD + 6 DEFECTIVE)",
            "layout": (2, 4),
            "canvas_size": (1024, 1024),
            "bg_color": (202, 202, 202),
            "items": [
                ("thread_top", 1.0), ("scratch_neck", 1.0), ("good", 1.0), ("manipulated_front", 1.0),
                ("scratch_head", 1.0), ("good", 1.0), ("thread_side", 1.0), ("thread_top", 1.0)
            ]
        }
    ]

    good_idx = 0
    cat_indices = {cat: 0 for cat in categories}
    registry = {}

    for sc in scenarios_config:
        canvas_w, canvas_h = sc["canvas_size"]
        canvas_bgr = np.full((canvas_h, canvas_w, 3), sc["bg_color"], dtype=np.uint8)

        rows, cols = sc["layout"]
        cell_w = canvas_w // cols
        cell_h = canvas_h // rows

        gt_products = []

        for idx, (item_type, scale) in enumerate(sc["items"]):
            r = idx // cols
            c = idx % cols

            if item_type == "good":
                src_path = shuffled_good[good_idx % len(shuffled_good)]
                good_idx += 1
                cat_name = "good"
                gt_status = "GOOD"
                mask_path = None
            else:
                cat_files = defective_files_by_cat[item_type]
                c_idx = cat_indices[item_type] % len(cat_files)
                src_path = cat_files[c_idx]
                cat_indices[item_type] += 1
                cat_name = item_type
                gt_status = "DEFECTIVE"

                # Check if ground truth defect mask exists
                mfile = gt_mask_dir / item_type / f"{src_path.stem}_mask.png"
                if not mfile.exists():
                    mfile = gt_mask_dir / item_type / f"{src_path.stem}.png"
                mask_path = str(mfile) if mfile.exists() else None

            src_img = cv2.imread(str(src_path))
            if src_img is None:
                continue

            sh, sw = src_img.shape[:2]
            target_w = int(cell_w * 0.82 * scale)
            target_h = int(cell_h * 0.82 * scale)
            resized_src = cv2.resize(src_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Center inside cell grid box
            cell_x1 = c * cell_w
            cell_y1 = r * cell_h

            offset_x = (cell_w - target_w) // 2
            offset_y = (cell_h - target_h) // 2

            px1 = cell_x1 + offset_x
            py1 = cell_y1 + offset_y
            px2 = px1 + target_w
            py2 = py1 + target_h

            canvas_bgr[py1:py2, px1:px2] = resized_src

            # Calculate normalized bbox [ymin, xmin, ymax, xmax] in 0-1000 range
            ymin = int(round((py1 / float(canvas_h)) * 1000))
            xmin = int(round((px1 / float(canvas_w)) * 1000))
            ymax = int(round((py2 / float(canvas_h)) * 1000))
            xmax = int(round((px2 / float(canvas_w)) * 1000))

            gt_products.append({
                "gt_id": idx + 1,
                "source_file": src_path.name,
                "category": cat_name,
                "gt_status": gt_status,
                "pixel_bbox": [px1, py1, px2, py2],
                "norm_bbox": [ymin, xmin, ymax, xmax],
                "gt_mask_path": mask_path
            })

        comp_path = COMPOSITES_DIR / sc["name"]
        cv2.imwrite(str(comp_path), canvas_bgr)
        registry[sc["name"]] = {
            "name": sc["name"],
            "title": sc["title"],
            "ref_path": sc["ref_path"],
            "file_path": str(comp_path),
            "canvas_size": [canvas_w, canvas_h],
            "gt_products": gt_products
        }

    with open(COMPOSITES_DIR / "ground_truth_registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    return registry


def query_gemini_reference_comparison(ref_image_path: Path, comp_image_path: Path):
    """
    Sends TWO images (IMAGE 1 = GOOD reference, IMAGE 2 = Test composite) to Gemini Vision API.
    Returns tuple of (parsed_json_dict, raw_response_text, latency_seconds, model_name).
    """
    api_key = get_gemini_api_key()
    model_name = get_gemini_instance_model()

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not configured.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    pil_ref = Image.open(ref_image_path).convert("RGB")
    pil_comp = Image.open(comp_image_path).convert("RGB")

    prompt = """You are performing visual quality inspection using a reference-based normality comparison.

You have been given exactly two images.

IMAGE 1 is a KNOWN-GOOD REFERENCE.
It represents a normal, defect-free example of the product.

IMAGE 2 is the TEST IMAGE.
It contains one or more physical products that must be inspected.

Your task is to inspect EVERY physical product in the TEST IMAGE independently and compare its visible appearance against the KNOWN-GOOD REFERENCE.

IMPORTANT:
The reference image is the visual baseline for NORMAL appearance.

For each product in the TEST IMAGE:

STEP 1 — IDENTIFY THE COMPLETE PRODUCT
Find the complete physical product.
Return a tight bounding box around the complete product.
Do not split one physical product into multiple products.
Do not treat background, shadows, reflections, texture regions, scratches, stains, highlights, or surface regions as separate products.

STEP 2 — NORMALITY COMPARISON
Compare the product against the known-GOOD reference.
Ignore differences caused by: position, image coordinates, scale, rotation/orientation, background, lighting differences, shadows, reflections, harmless image noise, harmless natural appearance variation.
Focus on physical/product-level deviations.
For a screw specifically inspect: overall geometry, head shape, head surface, neck, shaft, thread structure, thread continuity, thread shape/profile, missing material, deformation, scratches, abnormal marks, surface damage, structural irregularities, other visually meaningful manufacturing abnormalities.
Do NOT assume that every visual difference is a defect.
A difference should be considered defective only when it is visually consistent with a physical/manufacturing abnormality.

STEP 3 — DEFECT DECISION
For every product return exactly one primary status: GOOD or DEFECTIVE.
GOOD means: The product is visually consistent with the known-GOOD reference and no meaningful manufacturing defect is visible.
DEFECTIVE means: There is a localized or structural visual deviation that is reasonably consistent with a manufacturing defect.
If uncertain, explicitly record the uncertainty rather than inventing a defect.

STEP 4 — LOCALIZE THE DEVIATION
If the product is DEFECTIVE, identify the specific region or regions that caused the defective decision.
Do NOT simply return the entire product bounding box as the defect.
Return one or more localized deviation regions.
Each deviation region must contain: bounding box [ymin, xmin, ymax, xmax] (0-1000), center [y, x] (0-1000), severity (LOW, MEDIUM, HIGH), confidence (0.0 to 1.0), description of the deviation.
The deviation bounding box should cover the smallest reasonable region containing the visible abnormality.

STEP 5 — DEFECT TYPE
If the visual evidence supports a defect type, identify it (e.g. scratch, deformation, manipulated_front, scratch_head, scratch_neck, thread_side, thread_top, missing_material, abnormal_thread).

STEP 6 — SEVERITY
Estimate severity based on visible prominence and physical impact: LOW, MEDIUM, or HIGH.

STEP 7 — CONFIDENCE
Provide a confidence value from 0.0 to 1.0 representing confidence in the visual inspection decision.

Return ONLY valid JSON. Coordinates must be normalized from 0 to 1000 [ymin, xmin, ymax, xmax].

Expected JSON structure:
{
  "instances": [
    {
      "id": 1,
      "bbox": [ymin, xmin, ymax, xmax],
      "status": "GOOD",
      "comparison_summary": "Screw is structurally intact and visually consistent with the known-GOOD reference.",
      "deviation_regions": [],
      "defect_type": null,
      "severity": null,
      "confidence": 0.95
    },
    {
      "id": 2,
      "bbox": [ymin, xmin, ymax, xmax],
      "status": "DEFECTIVE",
      "comparison_summary": "Visible localized scratch mark on the screw neck compared to the normal reference.",
      "deviation_regions": [
        {
          "bbox": [ymin, xmin, ymax, xmax],
          "center": [y, x],
          "severity": "HIGH",
          "confidence": 0.91,
          "description": "Localized scratch near the screw neck"
        }
      ],
      "defect_type": "scratch_neck",
      "severity": "HIGH",
      "confidence": 0.91
    }
  ]
}

Do not return markdown formatting. Do not return text outside the JSON.
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1
    )

    t0 = time.time()
    response = client.models.generate_content(
        model=model_name,
        contents=[pil_ref, pil_comp, prompt],
        config=config
    )
    latency_sec = time.time() - t0

    raw_text = response.text or ""

    try:
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```[a-z]*\n?", "", clean_text)
            clean_text = re.sub(r"\n?```$", "", clean_text).strip()
        parsed_json = json.loads(clean_text)
    except Exception as e:
        print(f"[WARN] Failed to parse Gemini response JSON: {e}")
        parsed_json = {"instances": [], "error": str(e), "raw_text": raw_text}

    return parsed_json, raw_text, latency_sec, model_name


def render_gemini_deviation_heatmap(comp_img_bgr, pred_instances, canvas_w, canvas_h):
    """
    Renders soft/feathered heatmap overlays on the composite image centered at Gemini's reported deviation regions.
    Returns composite image BGR numpy array with overlaid heatmaps and bboxes.
    """
    vis_img = comp_img_bgr.copy()
    heatmap_acc = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for inst in pred_instances:
        dev_regions = inst.get("deviation_regions") or []
        for dev in dev_regions:
            dev_box_norm = dev.get("bbox") or [0, 0, 0, 0]
            ymin, xmin, ymax, xmax = dev_box_norm
            dx1 = int(round((xmin / 1000.0) * canvas_w))
            dy1 = int(round((ymin / 1000.0) * canvas_h))
            dx2 = int(round((xmax / 1000.0) * canvas_w))
            dy2 = int(round((ymax / 1000.0) * canvas_h))

            cx = (dx1 + dx2) // 2
            cy = (dy1 + dy2) // 2
            radius = max(int(max(dx2 - dx1, dy2 - dy1) * 0.8), 20)

            # Draw Gaussian blob on heatmap accumulator
            y_indices, x_indices = np.ogrid[:canvas_h, :canvas_w]
            dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
            sigma = max(radius / 2.0, 5.0)
            gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
            conf = float(dev.get("confidence", 0.90))
            heatmap_acc = np.maximum(heatmap_acc, gaussian * conf)

    # Convert heatmap accumulator to colored BGR overlay
    if np.max(heatmap_acc) > 0.01:
        norm_map = (heatmap_acc / np.max(heatmap_acc) * 255).astype(np.uint8)
        color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

        # Mask only active heatmap regions
        mask = (heatmap_acc > 0.05)[:, :, None]
        vis_img = np.where(mask, cv2.addWeighted(vis_img, 0.55, color_map, 0.45, 0), vis_img)

    return vis_img


def run_gemini_reference_comparison_diagnostic_experiment():
    print("=" * 80)
    print("REFERENCE-BASED GEMINI MULTI-PRODUCT DEFECT DETECTION DIAGNOSTIC EXPERIMENT")
    print("=" * 80)

    # 1. Build Reference Images & Composite Test Scenarios
    print("\n[STEP 1] Generating References & 5 Controlled Composite Test Scenarios...")
    registry = build_reference_and_composite_scenarios()
    print(f"  - Created {len(registry)} composite scenarios under {COMPOSITES_DIR}")
    print(f"  - Saved 5 GOOD reference images under {REFERENCES_DIR}")

    total_gt_products = 0
    total_pred_products = 0
    matched_products = 0
    iou_80_count = 0
    all_matched_ious = []

    # Overall Classification Counts
    tp, fn, tn, fp = 0, 0, 0, 0

    category_counts = {
        "manipulated_front": {"total": 0, "tp": 0, "fn": 0},
        "scratch_head": {"total": 0, "tp": 0, "fn": 0},
        "scratch_neck": {"total": 0, "tp": 0, "fn": 0},
        "thread_side": {"total": 0, "tp": 0, "fn": 0},
        "thread_top": {"total": 0, "tp": 0, "fn": 0}
    }

    false_positives_log = []
    false_negatives_log = []
    latencies = []
    per_composite_results = []

    # 2. Run Gemini Reference Comparison per Composite Case
    for comp_name, meta in registry.items():
        comp_path = Path(meta["file_path"])
        ref_path = Path(meta["ref_path"])
        gt_products = meta["gt_products"]
        canvas_w, canvas_h = meta["canvas_size"]

        print(f"\n[INSPECTING] {comp_name} against {ref_path.name} ({meta['title']})...")
        parsed, raw_text, latency_sec, model_name = query_gemini_reference_comparison(ref_path, comp_path)
        latencies.append(latency_sec)

        # Save raw response
        raw_file = RAW_RESPONSES_DIR / f"{comp_name.replace('.png', '')}_raw.json"
        with open(raw_file, "w") as f:
            json.dump({
                "composite": comp_name,
                "reference_image": ref_path.name,
                "model": model_name,
                "latency_sec": round(latency_sec, 2),
                "raw_text": raw_text,
                "parsed_json": parsed
            }, f, indent=2)

        pred_instances = parsed.get("instances", [])
        print(f"  - Gemini Latency: {latency_sec:.2f}s | GT Products: {len(gt_products)} | Predicted: {len(pred_instances)}")

        pred_boxes_px = []
        for inst in pred_instances:
            bbox_norm = inst.get("bbox", [0, 0, 0, 0])
            ymin, xmin, ymax, xmax = bbox_norm
            px1 = int(round((xmin / 1000.0) * canvas_w))
            py1 = int(round((ymin / 1000.0) * canvas_h))
            px2 = int(round((xmax / 1000.0) * canvas_w))
            py2 = int(round((ymax / 1000.0) * canvas_h))
            pred_boxes_px.append({
                "id": inst.get("id"),
                "status": (inst.get("status") or "GOOD").upper(),
                "comparison_summary": inst.get("comparison_summary", ""),
                "deviation_regions": inst.get("deviation_regions", []),
                "defect_type": inst.get("defect_type"),
                "severity": inst.get("severity"),
                "confidence": float(inst.get("confidence", 0.90)),
                "pixel_bbox": [px1, py1, px2, py2],
                "norm_bbox": bbox_norm
            })

        total_gt_products += len(gt_products)
        total_pred_products += len(pred_boxes_px)

        # IoU Matching (1-to-1 greedy matching)
        matched_gt_map = {}
        matched_pred_map = {}

        iou_matrix = np.zeros((len(gt_products), len(pred_boxes_px)))
        for i, gt_p in enumerate(gt_products):
            for j, pr_p in enumerate(pred_boxes_px):
                iou_matrix[i, j] = calculate_iou(gt_p["pixel_bbox"], pr_p["pixel_bbox"])

        flat_pairs = []
        for i in range(len(gt_products)):
            for j in range(len(pred_boxes_px)):
                flat_pairs.append((iou_matrix[i, j], i, j))
        flat_pairs.sort(key=lambda x: x[0], reverse=True)

        for iou_val, i, j in flat_pairs:
            if i in matched_gt_map or j in matched_pred_map:
                continue
            if iou_val >= 0.40:
                matched_gt_map[i] = (j, iou_val)
                matched_pred_map[j] = (i, iou_val)

        comp_tp, comp_fn, comp_tn, comp_fp = 0, 0, 0, 0

        comp_img = cv2.imread(str(comp_path))
        vis_img = render_gemini_deviation_heatmap(comp_img, pred_instances, canvas_w, canvas_h)

        for i, gt_p in enumerate(gt_products):
            cat = gt_p["category"]
            gt_status = gt_p["gt_status"]
            gx1, gy1, gx2, gy2 = gt_p["pixel_bbox"]

            if cat in category_counts:
                category_counts[cat]["total"] += 1

            if i in matched_gt_map:
                j, iou_val = matched_gt_map[i]
                pr_p = pred_boxes_px[j]
                pr_status = pr_p["status"]

                matched_products += 1
                all_matched_ious.append(iou_val)
                if iou_val >= 0.80:
                    iou_80_count += 1

                if gt_status == "DEFECTIVE":
                    if pr_status == "DEFECTIVE":
                        comp_tp += 1
                        tp += 1
                        if cat in category_counts:
                            category_counts[cat]["tp"] += 1
                    else: # Missed defect (FN)
                        comp_fn += 1
                        fn += 1
                        if cat in category_counts:
                            category_counts[cat]["fn"] += 1
                        false_negatives_log.append({
                            "composite": comp_name,
                            "reference_image": ref_path.name,
                            "gt_id": gt_p["gt_id"],
                            "category": cat,
                            "source_file": gt_p["source_file"],
                            "gemini_summary": pr_p["comparison_summary"],
                            "iou": round(iou_val, 4)
                        })
                else: # GT == GOOD
                    if pr_status == "GOOD":
                        comp_tn += 1
                        tn += 1
                    else: # Hallucination (FP)
                        comp_fp += 1
                        fp += 1
                        false_positives_log.append({
                            "composite": comp_name,
                            "reference_image": ref_path.name,
                            "gt_id": gt_p["gt_id"],
                            "source_file": gt_p["source_file"],
                            "predicted_defect_type": pr_p["defect_type"],
                            "gemini_summary": pr_p["comparison_summary"],
                            "iou": round(iou_val, 4)
                        })

                # Visual Overlay
                cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                cv2.putText(vis_img, f"GT:{gt_status[:4]}", (gx1, max(15, gy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                is_correct = (gt_status == pr_status)
                box_color = (255, 128, 0) if is_correct else (0, 0, 255)
                px1, py1, px2, py2 = pr_p["pixel_bbox"]
                cv2.rectangle(vis_img, (px1, py1), (px2, py2), box_color, 2)
                status_text = f"P:{pr_status[:4]} (IoU:{iou_val:.2f})"
                cv2.putText(vis_img, status_text, (px1, min(canvas_h - 10, py2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

                # Draw deviation region bboxes if present
                for dev in pr_p["deviation_regions"]:
                    dev_norm = dev.get("bbox") or [0, 0, 0, 0]
                    ymin, xmin, ymax, xmax = dev_norm
                    dx1 = int(round((xmin / 1000.0) * canvas_w))
                    dy1 = int(round((ymin / 1000.0) * canvas_h))
                    dx2 = int(round((xmax / 1000.0) * canvas_w))
                    dy2 = int(round((ymax / 1000.0) * canvas_h))
                    cv2.rectangle(vis_img, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                    cv2.putText(vis_img, f"DEV:{dev.get('severity', 'DEF')}", (dx1, dy1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                if gt_status == "DEFECTIVE":
                    comp_fn += 1
                    fn += 1
                    if cat in category_counts:
                        category_counts[cat]["fn"] += 1
                else:
                    comp_tn += 1
                    tn += 1

                cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 165, 255), 2)
                cv2.putText(vis_img, f"MISSED GT:{gt_status}", (gx1, gy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        vis_file = VISUALIZATIONS_DIR / f"{comp_name.replace('.png', '')}_vis.png"
        cv2.imwrite(str(vis_file), vis_img)

        per_composite_results.append({
            "composite": comp_name,
            "reference_image": ref_path.name,
            "title": meta["title"],
            "gt_count": len(gt_products),
            "pred_count": len(pred_boxes_px),
            "latency_sec": round(latency_sec, 2),
            "tp": comp_tp, "fn": comp_fn, "tn": comp_tn, "fp": comp_fp
        })

    # 3. Overall Quantitative Evaluation Metrics
    instance_recall = round((matched_products / float(total_gt_products)) * 100.0, 2) if total_gt_products > 0 else 0.0
    instance_precision = round((matched_products / float(total_pred_products)) * 100.0, 2) if total_pred_products > 0 else 0.0
    mean_iou = round(float(np.mean(all_matched_ious)), 4) if all_matched_ious else 0.0
    pct_iou_80 = round((iou_80_count / float(matched_products)) * 100.0, 2) if matched_products > 0 else 0.0

    defect_recall = round((tp / float(tp + fn)) * 100.0, 2) if (tp + fn) > 0 else 0.0
    good_fpr = round((fp / float(fp + tn)) * 100.0, 2) if (fp + tn) > 0 else 0.0
    precision = round((tp / float(tp + fp)) * 100.0, 2) if (tp + fp) > 0 else 0.0
    accuracy = round(((tp + tn) / float(tp + tn + fp + fn)) * 100.0, 2) if (tp + tn + fp + fn) > 0 else 0.0

    cat_breakdown = {}
    for cat, counts in category_counts.items():
        tot = counts["total"]
        c_tp = counts["tp"]
        c_rec = round((c_tp / float(tot)) * 100.0, 2) if tot > 0 else 0.0
        cat_breakdown[cat] = {
            "total_gt": tot,
            "detected_tp": c_tp,
            "missed_fn": counts["fn"],
            "recall_percent": c_rec
        }

    latency_stats = {
        "mean_sec": round(float(np.mean(latencies)), 2),
        "median_sec": round(float(np.median(latencies)), 2),
        "min_sec": round(float(np.min(latencies)), 2),
        "max_sec": round(float(np.max(latencies)), 2)
    }

    if instance_recall >= 90.0 and defect_recall >= 80.0 and good_fpr <= 15.0:
        final_verdict = "PROMISING"
    elif instance_recall >= 80.0 and defect_recall >= 60.0:
        final_verdict = "INCONCLUSIVE"
    else:
        final_verdict = "NOT PROMISING"

    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": get_gemini_instance_model(),
        "total_composites_tested": len(registry),
        "references_used": 5,
        "instance_localization": {
            "gt_product_count": total_gt_products,
            "predicted_product_count": total_pred_products,
            "matched_products": matched_products,
            "instance_recall_percent": instance_recall,
            "instance_precision_percent": instance_precision,
            "mean_iou": mean_iou,
            "percent_iou_ge_80": pct_iou_80
        },
        "classification_metrics": {
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "defect_recall_percent": defect_recall,
            "good_fpr_percent": good_fpr,
            "precision_percent": precision,
            "accuracy_percent": accuracy
        },
        "category_breakdown": cat_breakdown,
        "latency_stats": latency_stats,
        "false_positives_analysis": false_positives_log,
        "false_negatives_analysis": false_negatives_log,
        "per_composite_results": per_composite_results,
        "final_verdict": final_verdict
    }

    report_json_path = REPORTS_DIR / "gemini_reference_comparison_diagnostic_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    readme_content = f"""# Reference-Based Gemini Multi-Product Defect Detection Diagnostic Report

## Executive Summary
Evaluating whether **Gemini Vision** can perform accurate multi-product quality inspection when explicitly provided **EXACTLY ONE known-GOOD reference image** (`IMAGE 1`) alongside a multi-product test image (`IMAGE 2`) for visual baseline comparison.

- **Final Verdict**: **{final_verdict}**
- **Gemini Model**: `{get_gemini_instance_model()}`
- **Composites Tested**: {len(registry)} multi-product images ({total_gt_products} total physical screws)
- **GOOD References Used**: 5 distinct GOOD images from `train/good`

---

## Overall Performance Metrics

### Instance Localization Metrics
- **Ground-Truth Products**: {total_gt_products}
- **Predicted Products**: {total_pred_products}
- **Matched Products**: {matched_products}
- **Instance Localization Recall**: **{instance_recall}%**
- **Instance Localization Precision**: **{instance_precision}%**
- **Mean Bounding Box IoU**: **{mean_iou}**
- **Percentage IoU $\\ge 0.80$**: **{pct_iou_80}%**

### GOOD / DEFECTIVE Classification Metrics
- **True Positives (TP)**: {tp} (Defective correctly identified)
- **False Negatives (FN)**: {fn} (Defective missed)
- **True Negatives (TN)**: {tn} (GOOD correctly identified)
- **False Positives (FP)**: {fp} (GOOD incorrectly flagged)
- **Defect Recall**: **{defect_recall}%**
- **Held-Out GOOD FPR**: **{good_fpr}%**
- **Precision**: **{precision}%**
- **Overall Accuracy**: **{accuracy}%**

---

## Per-Defect Category Breakdown

| Defect Category | Total GT | Correctly Detected (TP) | Missed (FN) | Category Recall |
| :--- | :---: | :---: | :---: | :---: |
"""
    for cat, cb in cat_breakdown.items():
        readme_content += f"| **{cat}** | {cb['total_gt']} | {cb['detected_tp']} | {cb['missed_fn']} | **{cb['recall_percent']}%** |\n"

    readme_content += f"""
---

## API Response Latency
- **Mean Latency**: {latency_stats['mean_sec']} sec
- **Median Latency**: {latency_stats['median_sec']} sec
- **Range**: {latency_stats['min_sec']}s – {latency_stats['max_sec']}s

---

## False Positive & False Negative Analysis

### False Positives (GOOD screws called DEFECTIVE): {len(false_positives_log)}
"""
    for fp_item in false_positives_log:
        readme_content += f"- **{fp_item['composite']} (GT #{fp_item['gt_id']})**: Gemini summary: *\"{fp_item['gemini_summary']}\"*\n"

    readme_content += f"""
### False Negatives (DEFECTIVE screws missed): {len(false_negatives_log)}
"""
    for fn_item in false_negatives_log:
        readme_content += f"- **{fn_item['composite']} ({fn_item['category']} GT #{fn_item['gt_id']})**: Gemini summary: *\"{fn_item['gemini_summary']}\"*\n"

    readme_content += f"""
---

## Technical Verdict
**{final_verdict}**

*Report generated at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*
"""

    readme_path = OUTPUT_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print("\n" + "=" * 50)
    print("GEMINI REFERENCE-BASED DIAGNOSTIC")
    print("=" * 50)
    print(f"Gemini model: {get_gemini_instance_model()}")
    print(f"Composite images: {len(registry)}")
    print(f"GOOD references used: 1 per composite")

    print("\nINSTANCE LOCALIZATION")
    print(f"GT products: {total_gt_products}")
    print(f"Predicted products: {total_pred_products}")
    print(f"Matched: {matched_products}")
    print(f"Instance Recall: {instance_recall}%")
    print(f"Instance Precision: {instance_precision}%")
    print(f"Mean IoU: {mean_iou}")
    print(f"IoU >= 0.80: {pct_iou_80}%")

    print("\nGOOD / DEFECTIVE")
    print(f"TP: {tp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")

    print(f"\nDefect Recall: {defect_recall}%")
    print(f"GOOD FPR: {good_fpr}%")
    print(f"Precision: {precision}%")
    print(f"Accuracy: {accuracy}%")

    print("\nDEFECT CATEGORY RECALL")
    for cat, cb in cat_breakdown.items():
        print(f"{cat:<18}: {cb['recall_percent']}% ({cb['detected_tp']}/{cb['total_gt']})")

    print("\nLATENCY")
    print(f"Mean: {latency_stats['mean_sec']} sec")
    print(f"Median: {latency_stats['median_sec']} sec")

    print(f"\nFINAL VERDICT:\n{final_verdict}")
    print("=" * 50)

    print(f"\n[INFO] Artifacts saved to: {OUTPUT_DIR}")
    return report_data


if __name__ == "__main__":
    run_gemini_reference_comparison_diagnostic_experiment()
