"""
evaluate_rd4ad.py

Threshold/CC/morphology sweep on ReverseDistillation (RD4AD) predictions.
Gate: Recall >= 90%, Mean IoU >= 80%, FPR < 5%
"""

from pathlib import Path
import json
import csv
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import ReverseDistillation

# ── Config ────────────────────────────────────────────────────
BACKBONE  = "wide_resnet50_2"
CATEGORY  = "screw"
SEED      = 42

BASE_DIR        = Path(__file__).resolve().parents[1]
DATASET_ROOT    = BASE_DIR / "data" / "mvtec_anomaly_detection"
CHECKPOINT      = BASE_DIR / "outputs" / "rd4ad_baseline" / f"rd4ad_screw_{BACKBONE}.ckpt"
OUTPUT_DIR      = BASE_DIR / "outputs" / "rd4ad_baseline" / f"screw_{BACKBONE}_localization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("=" * 70)
print("RD4AD LOCALIZATION EVALUATION")
print("=" * 70)
print(f"Checkpoint: {CHECKPOINT}")
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Run train_rd4ad.py first.\nMissing: {CHECKPOINT}")

# ── Inference ────────────────────────────────────────────────
print("\n[1/5] Running inference ...")
datamodule = MVTecAD(root=str(DATASET_ROOT), category=CATEGORY, eval_batch_size=4, num_workers=0)
datamodule.setup()

from anomalib.models.image.reverse_distillation.anomaly_map import AnomalyMapGenerationMode

model = ReverseDistillation(
    backbone=BACKBONE, pre_trained=True,
    layers=["layer1", "layer2", "layer3"],
    anomaly_map_mode=AnomalyMapGenerationMode.MULTIPLY,
)

# Ensure inference resolution matches exact 256x256 training resolution
orig_predict_step = model.predict_step
def custom_predict_step(batch, batch_idx, dataloader_idx=0):
    if hasattr(batch, "image") and isinstance(batch.image, torch.Tensor):
        if batch.image.shape[-2:] != (256, 256):
            batch.image = torch.nn.functional.interpolate(batch.image, size=(256, 256), mode="bilinear", align_corners=False)
    return orig_predict_step(batch, batch_idx, dataloader_idx)
model.predict_step = custom_predict_step

engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)
prediction_batches = engine.predict(model=model, datamodule=datamodule,
                                    ckpt_path=str(CHECKPOINT))
if not prediction_batches:
    raise RuntimeError("engine.predict() returned nothing.")
print(f"Got {len(prediction_batches)} batch(es).")

# ── Extract samples ──────────────────────────────────────────
print("\n[2/5] Extracting samples ...")

def to_np(v):
    if v is None: return None
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

samples = []
for batch in prediction_batches:
    paths = getattr(batch, "image_path", None)
    if paths is None: continue
    images     = to_np(getattr(batch, "image",       None))
    amaps      = to_np(getattr(batch, "anomaly_map", None))
    pred_masks = to_np(getattr(batch, "pred_mask",   None))
    gt_masks   = to_np(getattr(batch, "gt_mask",     None))
    gt_labels  = to_np(getattr(batch, "gt_label",    None))

    for i, path in enumerate(paths):
        path = Path(path)
        def squeeze(arr, idx):
            if arr is None: return None
            a = arr[idx]
            return a[0] if a.ndim == 3 and a.shape[0] == 1 else a
        img = images[i] if images is not None else None
        if img is not None and img.ndim == 3 and img.shape[0] in [1,3]:
            img = np.transpose(img, (1,2,0))
        
        amap = squeeze(amaps, i)
        pred_m = squeeze(pred_masks, i)
        gt_m = squeeze(gt_masks, i)
        
        # Resize to 256x256 to match 256x256 training resolution
        if img is not None and img.shape[:2] != (256, 256):
            img = cv2.resize(img, (256, 256))
        if amap is not None and amap.shape[:2] != (256, 256):
            amap = cv2.resize(amap, (256, 256))
        if pred_m is not None and pred_m.shape[:2] != (256, 256):
            pred_m = cv2.resize(pred_m.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
        if gt_m is not None and gt_m.shape[:2] != (256, 256):
            gt_m = cv2.resize(gt_m.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)

        samples.append({
            "image_path":         str(path),
            "defect_type":        path.parent.name,
            "image":              img,
            "anomaly_map":        amap,
            "baseline_pred_mask": pred_m,
            "gt_mask":            gt_m,
            "gt_label":           bool(gt_labels[i]) if gt_labels is not None else (path.parent.name != "good"),
        })

print(f"Samples: {len(samples)}")
for cat, cnt in sorted({s["defect_type"]: 0 for s in samples}.items()):
    cnt = sum(1 for s in samples if s["defect_type"] == cat)
    print(f"  {cat:<25}: {cnt}")

# ── 50/50 stratified split ────────────────────────────────────
print("\n[3/5] Stratified val/test split (seed=42) ...")
by_cat = {}
for s in samples:
    by_cat.setdefault(s["defect_type"], []).append(s)

val_samples = []; test_samples = []; split_info = {}
for cat, items in sorted(by_cat.items()):
    random.seed(SEED); random.shuffle(items)
    n = len(items) // 2
    val_samples.extend(items[:n]); test_samples.extend(items[n:])
    split_info[cat] = {"total": len(items), "val_count": n, "test_count": len(items)-n}

# Normalize anomaly maps using val-set percentiles (no test leakage)
all_val_maps = [s["anomaly_map"] for s in val_samples if s["anomaly_map"] is not None]
g_min = float(min(m.min() for m in all_val_maps))
g_max = float(max(m.max() for m in all_val_maps))
print(f"Anomaly map range (val): [{g_min:.4f}, {g_max:.4f}]")

for s in samples:
    if s["anomaly_map"] is not None:
        s["norm_map"] = np.clip(
            (s["anomaly_map"] - g_min) / (g_max - g_min + 1e-8), 0.0, 1.0
        ).astype(np.float32)

# ── Helpers ───────────────────────────────────────────────────
def sample_metrics(gt_mask, pred_mask, is_defective):
    has_pred = bool(np.any(pred_mask > 0)) if pred_mask is not None else False
    tp = fn = good_fp = 0
    if is_defective:
        tp, fn = (1, 0) if has_pred else (0, 1)
    else:
        good_fp = 1 if has_pred else 0
    iou = None
    if gt_mask is not None and pred_mask is not None:
        gt   = np.asarray(gt_mask).astype(bool)
        pred = np.asarray(pred_mask).astype(bool)
        inter = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        iou = float(inter/union) if union > 0 else (1.0 if inter == 0 else 0.0)
    return tp, fn, good_fp, iou

def post_process(norm_map, threshold, min_cc=0, morph=None):
    if norm_map is None: return None
    b = (norm_map >= threshold).astype(np.uint8)
    if min_cc > 0 and np.any(b):
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        f = np.zeros_like(b)
        for lbl_id in range(1, n):
            if stats[lbl_id, cv2.CC_STAT_AREA] >= min_cc:
                f[lbl == lbl_id] = 1
        b = f
    if morph is not None and np.any(b):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        if morph == "open_close":
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
        elif morph == "opening":
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
        elif morph == "closing":
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
        elif morph == "dilate":
            b = cv2.morphologyEx(b, cv2.MORPH_DILATE, k)
    return b

def eval_set(sample_list, threshold, min_cc=0, morph=None, use_baseline=False):
    total_tp = total_fn = good_total = good_fp = 0
    ious = []; per_defect = {}
    for s in sample_list:
        is_def = s["gt_label"]; cat = s["defect_type"]
        if use_baseline:
            pred_m = (s["baseline_pred_mask"] > 0).astype(np.uint8) \
                     if s["baseline_pred_mask"] is not None else None
        else:
            pred_m = post_process(s.get("norm_map"), threshold, min_cc, morph)
        tp, fn, gfp, iou = sample_metrics(s["gt_mask"], pred_m, is_def)
        if is_def:
            total_tp += tp; total_fn += fn
            per_defect.setdefault(cat, {"total":0,"tp":0,"fn":0,"ious":[]})
            per_defect[cat]["total"] += 1
            per_defect[cat]["tp"]    += tp
            per_defect[cat]["fn"]    += fn
            if iou is not None:
                per_defect[cat]["ious"].append(iou); ious.append(iou)
        else:
            good_total += 1; good_fp += gfp
    return {
        "recall":   (total_tp/(total_tp+total_fn))*100 if (total_tp+total_fn)>0 else 0,
        "mean_iou": float(np.mean(ious))*100 if ious else 0,
        "fpr":      (good_fp/good_total)*100 if good_total>0 else 0,
        "tp":total_tp, "fn":total_fn, "good_total":good_total, "good_fp":good_fp,
        "per_defect": per_defect,
    }

# ── Sweeps ─────────────────────────────────────────────────────
print("\n[4/5] Optimization sweeps on validation set ...")

thresholds = [round(t, 2) for t in
              [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,
               0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]]

# A — Threshold
thresh_rows = []; best_thresh = thresholds[0]; best_score_t = (-1,-1,-1.)
for t in thresholds:
    r = eval_set(val_samples, t)
    thresh_rows.append({"threshold":t,"recall":r["recall"],"mean_iou":r["mean_iou"],
                         "fpr":r["fpr"],"tp":r["tp"],"fn":r["fn"],"good_fp":r["good_fp"]})
    sk = (1 if r["recall"]>=90 else 0, 1 if r["fpr"]<5 else 0, r["mean_iou"])
    if sk > best_score_t: best_score_t = sk; best_thresh = t

with open(OUTPUT_DIR/"threshold_sweep.csv","w",newline="") as f:
    w = csv.DictWriter(f, thresh_rows[0].keys()); w.writeheader(); w.writerows(thresh_rows)

print(f"\n{'Thresh':>8} {'Recall':>8} {'IoU':>8} {'FPR':>8}")
for row in thresh_rows:
    flag = " <-- BEST" if row["threshold"]==best_thresh else ""
    print(f"  {row['threshold']:>6.2f}  {row['recall']:>7.2f}%  {row['mean_iou']:>7.2f}%  {row['fpr']:>7.2f}%{flag}")
print(f"\n-> Best Threshold: {best_thresh}  IoU: {best_score_t[2]:.2f}%")

# B — Connected Components
cc_sizes = [0,10,25,50,100,200,500]
cc_rows=[]; best_cc=0; best_score_cc=(-1,-1,-1.)
for cc in cc_sizes:
    r = eval_set(val_samples, best_thresh, min_cc=cc)
    cc_rows.append({"min_cc":cc,"recall":r["recall"],"mean_iou":r["mean_iou"],
                    "fpr":r["fpr"],"tp":r["tp"],"fn":r["fn"],"good_fp":r["good_fp"]})
    sk = (1 if r["recall"]>=90 else 0, 1 if r["fpr"]<5 else 0, r["mean_iou"])
    if sk > best_score_cc: best_score_cc = sk; best_cc = cc

with open(OUTPUT_DIR/"connected_component_sweep.csv","w",newline="") as f:
    w = csv.DictWriter(f, cc_rows[0].keys()); w.writeheader(); w.writerows(cc_rows)
print(f"-> Best CC size: {best_cc}  IoU: {best_score_cc[2]:.2f}%")

# C — Morphology
morphs = [None,"opening","closing","open_close","dilate"]
morph_rows=[]; best_morph=None; best_score_m=(-1,-1,-1.)
for mop in morphs:
    r = eval_set(val_samples, best_thresh, min_cc=best_cc, morph=mop)
    morph_rows.append({"morph":str(mop),"recall":r["recall"],"mean_iou":r["mean_iou"],
                       "fpr":r["fpr"],"tp":r["tp"],"fn":r["fn"],"good_fp":r["good_fp"]})
    sk = (1 if r["recall"]>=90 else 0, 1 if r["fpr"]<5 else 0, r["mean_iou"])
    if sk > best_score_m: best_score_m = sk; best_morph = mop

with open(OUTPUT_DIR/"morphology_sweep.csv","w",newline="") as f:
    w = csv.DictWriter(f, morph_rows[0].keys()); w.writeheader(); w.writerows(morph_rows)
print(f"-> Best Morphology: {best_morph}  IoU: {best_score_m[2]:.2f}%")

# D — Combo
cands_t = sorted(set([best_thresh,
                       max(0.05, round(best_thresh-0.05, 2)),
                       min(0.90, round(best_thresh+0.05, 2))]))
cands_cc    = sorted(set([best_cc, 0, 25, 50]))
cands_morph = [None,"opening","closing","open_close","dilate"]

combo_rows=[]; best_combo: dict = {"threshold":best_thresh,"min_cc":best_cc,"morph":best_morph}
best_score_combo = (-1,-1,-1.)
for t in cands_t:
    for cc in cands_cc:
        for mop in cands_morph:
            r = eval_set(val_samples, t, min_cc=cc, morph=mop)
            combo_rows.append({"threshold":t,"min_cc":cc,"morph":str(mop),
                               "recall":r["recall"],"mean_iou":r["mean_iou"],"fpr":r["fpr"],
                               "tp":r["tp"],"fn":r["fn"],"good_fp":r["good_fp"]})
            sk = (1 if r["recall"]>=90 else 0, 1 if r["fpr"]<5 else 0, r["mean_iou"])
            if sk > best_score_combo:
                best_score_combo = sk
                best_combo = {"threshold":t, "min_cc":cc, "morph":mop}

with open(OUTPUT_DIR/"combination_search.csv","w",newline="") as f:
    w = csv.DictWriter(f, combo_rows[0].keys()); w.writeheader(); w.writerows(combo_rows)

print(f"\nLocked config: {best_combo}")
with open(OUTPUT_DIR/"selected_validation_config.json","w") as f:
    json.dump(best_combo, f, indent=2)

# ── Final test evaluation ─────────────────────────────────────
print("\n[5/5] Final held-out test evaluation ...")

base_eval = eval_set(test_samples, threshold=0.5, use_baseline=True)
opt_eval  = eval_set(test_samples,
                     threshold=best_combo["threshold"],
                     min_cc=best_combo["min_cc"],
                     morph=best_combo["morph"])

def per_defect_fmt(pd):
    return {k: {"total":v["total"],"tp":v["tp"],"fn":v["fn"],
                "recall":(v["tp"]/v["total"])*100 if v["total"]>0 else 0,
                "mean_iou":float(np.mean(v["ious"]))*100 if v["ious"] else 0}
            for k,v in pd.items()}

summary = {
    "model":"ReverseDistillation","backbone":BACKBONE,
    "split_info":split_info, "selected_config":best_combo,
    "baseline": {**{k:base_eval[k] for k in ["recall","mean_iou","fpr","tp","fn","good_total","good_fp"]},
                 "per_defect": per_defect_fmt(base_eval["per_defect"])},
    "optimized": {**{k:opt_eval[k] for k in ["recall","mean_iou","fpr","tp","fn","good_total","good_fp"]},
                  "per_defect": per_defect_fmt(opt_eval["per_defect"])},
    "improvements": {
        "recall_diff":   opt_eval["recall"]   - base_eval["recall"],
        "mean_iou_diff": opt_eval["mean_iou"] - base_eval["mean_iou"],
        "fpr_diff":      opt_eval["fpr"]       - base_eval["fpr"],
    },
    "prd_gate": {
        "recall_pass":   opt_eval["recall"]   >= 90.0,
        "mean_iou_pass": opt_eval["mean_iou"] >= 80.0,
        "fpr_pass":      opt_eval["fpr"]       < 5.0,
        "overall_pass":  (opt_eval["recall"]>=90 and opt_eval["mean_iou"]>=80 and opt_eval["fpr"]<5),
    },
    "vs_patchcore": {
        "patchcore_iou":  24.68, "rd4ad_iou": opt_eval["mean_iou"],
        "improvement":    opt_eval["mean_iou"] - 24.68,
        "patchcore_recall": 86.89, "rd4ad_recall": opt_eval["recall"],
    }
}

with open(OUTPUT_DIR/"final_heldout_test_results.json","w") as f:
    json.dump(summary, f, indent=2)

# ── Print results ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("RD4AD FINAL RESULTS")
print("=" * 70)
print(f"\n{'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'PatchCore':>12}")
print("-" * 60)
print(f"{'Recall':<20} {base_eval['recall']:>11.2f}% {opt_eval['recall']:>11.2f}% {'86.89':>11}%")
print(f"{'Mean IoU':<20} {base_eval['mean_iou']:>11.2f}% {opt_eval['mean_iou']:>11.2f}% {'24.68':>11}%")
print(f"{'FPR':<20} {base_eval['fpr']:>11.2f}% {opt_eval['fpr']:>11.2f}% {'4.76':>11}%")

gate = summary["prd_gate"]
print("\nPRD GATE:")
print(f"  Recall >= 90%:   {'[PASS]' if gate['recall_pass']   else '[FAIL]'} ({opt_eval['recall']:.2f}%)")
print(f"  Mean IoU >= 80%: {'[PASS]' if gate['mean_iou_pass'] else '[FAIL]'} ({opt_eval['mean_iou']:.2f}%)")
print(f"  FPR < 5%:        {'[PASS]' if gate['fpr_pass']      else '[FAIL]'} ({opt_eval['fpr']:.2f}%)")
print(f"  OVERALL:         {'[ALL PASS]' if gate['overall_pass'] else '[NOT YET]'}")

print("\nPER-DEFECT (Optimized, Test Set):")
print(f"  {'Category':<25} {'Recall':>8} {'IoU':>10}")
print("  " + "-" * 46)
for cat, d in summary["optimized"]["per_defect"].items():
    print(f"  {cat:<25} {d['recall']:>7.2f}%  {d['mean_iou']:>8.2f}%")

print(f"\nSaved to: {OUTPUT_DIR}")
print("=" * 70)

# ── Visualizations ────────────────────────────────────────────
print("\nGenerating visualizations ...")
cats = ["manipulated_front","scratch_head","scratch_neck","thread_side","thread_top","good"]
for cat in cats:
    cat_s = [s for s in test_samples if s["defect_type"]==cat]
    if not cat_s: continue
    if cat=="good":
        fp_s = [s for s in cat_s
                if sample_metrics(s["gt_mask"],
                                  post_process(s.get("norm_map"),
                                               best_combo["threshold"],
                                               best_combo["min_cc"],
                                               best_combo["morph"]), False)[2]>0]
        vis = fp_s[:2] if fp_s else cat_s[:2]
    else:
        vis = cat_s[:1]

    for idx, s in enumerate(vis):
        fig, axes = plt.subplots(1, 5, figsize=(20,4))
        img = s["image"]
        if img is not None:
            axes[0].imshow((img*255).astype(np.uint8) if img.max()<=1 else img.astype(np.uint8))
        axes[0].set_title("1. Original"); axes[0].axis("off")

        nm = s.get("norm_map")
        if nm is not None:
            im = axes[1].imshow(nm, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].set_title("2. RD4AD Anomaly Map"); axes[1].axis("off")

        gt = s["gt_mask"]
        axes[2].imshow(gt if gt is not None else np.zeros((256,256)), cmap="gray")
        axes[2].set_title("3. GT Mask"); axes[2].axis("off")

        bp = s["baseline_pred_mask"]
        axes[3].imshow(bp if bp is not None else np.zeros((256,256)), cmap="gray")
        axes[3].set_title("4. Baseline Mask"); axes[3].axis("off")

        op = post_process(nm, best_combo["threshold"], best_combo["min_cc"], best_combo["morph"])
        axes[4].imshow(op if op is not None else np.zeros((256,256)), cmap="gray")
        _, _, _, iou_v = sample_metrics(gt, op, s["gt_label"])
        axes[4].set_title(f"5. Optimized (IoU={iou_v*100:.1f}%)" if iou_v else "5. Optimized"); axes[4].axis("off")

        plt.suptitle(f"RD4AD — {cat} sample {idx+1}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fname = f"vis_{cat}_sample{idx+1}.png"
        plt.savefig(OUTPUT_DIR/fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> {fname}")

print("\nRD4AD Evaluation Complete!")
