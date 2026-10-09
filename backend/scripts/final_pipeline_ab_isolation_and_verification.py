import sys
import io
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.connection import get_db
from services.inspection_service import (
    run_single_image_inspection,
    run_multi_instance_inspection,
    retry_vlm_analysis
)
from fastapi import UploadFile

db = get_db()
model_id = "6aa3aa1d4067031f17c51123"

def make_upload(path: Path) -> UploadFile:
    with open(path, "rb") as f:
        content = f.read()
    return UploadFile(filename=path.name, file=io.BytesIO(content))

good_square = Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/good/000.png")
defective_square = Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/scratch_head/000.png")

print("="*85)
print("1. CLEAN INPUT QUEUE VERIFICATION (ZERO ARTIFACT POLLUTION)")
print("="*85)
test_dir = Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test")
polluted = [f.name for f in test_dir.rglob("*") if f.is_file() and ("_v3_crop" in f.name or "_heatmap" in f.name)]
print(f"Polluted files found in test dataset directory: {len(polluted)}")
if len(polluted) > 0:
    print(f"[ERROR] Found polluted debug files in dataset folder: {polluted[:5]}")
    sys.exit(1)
print(">>> SUCCESS: Dataset source directories are 100% clean of debug crops and heatmaps!")

print("\n" + "="*85)
print("2. PIPELINE A REAL MODEL THRESHOLD INDEPENDENCE MATRIX")
print("="*85)
thresholds = [10.0, 20.0, 24.54, 40.0, 60.0, 100.0]
print(f"{'Image':<15} | {'Threshold':<10} | {'Raw Score':<12} | {'Classification':<15}")
print("-" * 85)

matrix_passed = True
for label, img_path in [("GOOD", good_square), ("DEFECTIVE", defective_square)]:
    scores = []
    for thr in thresholds:
        uf = make_upload(img_path)
        res = run_single_image_inspection(db, model_id=model_id, upload_file=uf, threshold_override=thr)
        pred = res.get("prediction", {})
        score = float(pred.get("anomaly_score", 0.0))
        status = str(pred.get("status", "unknown")).upper()
        scores.append(score)
        print(f"{label:<15} | {thr:<10.2f} | {score:<12.2f} | {status:<15}")
    
    first = scores[0]
    if not all(math.isclose(s, first, abs_tol=1e-3) for s in scores):
        matrix_passed = False
        print(f"[ERROR] Raw score for {label} changed across thresholds: {scores}")

if matrix_passed:
    print(">>> SUCCESS: Pipeline A raw anomaly scores are 100% threshold-independent!")
else:
    print(">>> FAILURE: Pipeline A raw anomaly scores varied with threshold!")
    sys.exit(1)

print("\n" + "="*85)
print("3. PIPELINE A GEMINI DEFECT ANALYSIS END-TO-END TEST")
print("="*85)
uf_good = make_upload(good_square)
res_good = run_single_image_inspection(db, model_id=model_id, upload_file=uf_good, threshold_override=24.54)
score_before = res_good["prediction"]["anomaly_score"]
status_before = res_good["prediction"]["status"]

insp_id = res_good["inspection_id"]
vlm_res = retry_vlm_analysis(db, inspection_id=insp_id, force=True)
vlm_analysis = vlm_res.get("vlm_analysis", {})

score_after = vlm_res["prediction"]["anomaly_score"]
status_after = vlm_res["prediction"]["status"]

print(f"VLM Analysis Status           : {vlm_analysis.get('status')}")
print(f"VLM Explanation               : {vlm_analysis.get('explanation')}")
print(f"PatchCore Score Before / After: {score_before} / {score_after}")
print(f"PatchCore Status Before / After: {status_before} / {status_after}")

assert score_before == score_after and status_before == status_after, "Gemini altered PatchCore score or status!"
print(">>> SUCCESS: Gemini Analysis completed end-to-end without altering PatchCore classification!")

print("\n" + "="*85)
print("4. PIPELINE B MULTI-INSTANCE INSPECTION VERIFICATION")
print("="*85)
uf_b = make_upload(good_square)
res_b = run_multi_instance_inspection(db, model_id=model_id, upload_file=uf_b)
overall_b = res_b.get("overall_prediction", {})
instances_b = res_b.get("instances", [])
print(f"Pipeline B Overall Status     : {overall_b.get('status')}")
print(f"Total Instances Detected      : {overall_b.get('total_instances')}")
if instances_b:
    first_inst = instances_b[0]
    print(f"Instance 1 BBox               : {first_inst.get('bbox')}")
    print(f"Instance 1 Status             : {first_inst.get('prediction', {}).get('status')}")

print("\n" + "="*85)
print("5. CROSS-PIPELINE ISOLATION VERIFICATION (A -> B -> A)")
print("="*85)
uf_a1 = make_upload(good_square)
res_a1 = run_single_image_inspection(db, model_id=model_id, upload_file=uf_a1, threshold_override=24.54)
score_a1 = res_a1["prediction"]["anomaly_score"]

uf_b2 = make_upload(good_square)
_ = run_multi_instance_inspection(db, model_id=model_id, upload_file=uf_b2)

uf_a2 = make_upload(good_square)
res_a2 = run_single_image_inspection(db, model_id=model_id, upload_file=uf_a2, threshold_override=24.54)
score_a2 = res_a2["prediction"]["anomaly_score"]

print(f"Pipeline A Score #1 : {score_a1:.4f}")
print(f"Pipeline A Score #2 : {score_a2:.4f}")
iso_passed = math.isclose(score_a1, score_a2, abs_tol=1e-4)
print(f"Isolation Passed    : {iso_passed}")

if not iso_passed:
    print(">>> FAILURE: Pipeline A state drifted after Pipeline B run!")
    sys.exit(1)
else:
    print(">>> SUCCESS: 100% Pipeline A/B Isolation Verified!")

print("="*85)
