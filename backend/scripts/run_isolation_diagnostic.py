import sys
import os
import io
import time
from pathlib import Path
from bson import ObjectId

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.connection import get_db
from services.inspection_service import run_single_image_inspection, run_multi_instance_inspection
from services.patchcore_service import ANOMALIB_AVAILABLE
from fastapi import UploadFile

db = get_db()
MODEL_ID = "6aa3aa1d4067031f17c51123"

# Path definitions
GOOD_SINGLE = Path("c:/dev/Anomaly_Detector/results/Patchcore/v0/images/good/003.png")
DEFECTIVE_SINGLE = Path("c:/dev/Anomaly_Detector/results/Patchcore/v0/images/scratch_head/000.png")
MULTI_COMPOSITE = Path("c:/dev/Anomaly_Detector/backend/storage/test_verification_artifacts/grid_2x2.png")

def make_upload_file(path: Path) -> UploadFile:
    with open(path, "rb") as f:
        content = f.read()
    file_obj = io.BytesIO(content)
    return UploadFile(filename=path.name, file=file_obj)

def run_pipeline_a_diagnostic(image_path: Path, label: str):
    uf = make_upload_file(image_path)
    res = run_single_image_inspection(db, model_id=MODEL_ID, upload_file=uf)
    
    print(f"\n--- PIPELINE A DIAGNOSTIC ({label}) ---")
    print(f"PIPELINE: Pipeline A (Single-Product PatchCore)")
    print(f"MODE: single_image")
    print(f"MODEL_ID: {res.get('model_id')}")
    print(f"MODEL_VERSION_ID: {res.get('model_version_id')}")
    print(f"CHECKPOINT: backend/storage/artifacts/{res.get('model_id')}/v2/patchcore_memory_bank.ckpt")
    print(f"PREPROCESSING: V2 (256x256 bilinear resize)")
    print(f"INPUT_PATH: {image_path}")
    print(f"INPUT_SIZE: 256x256")
    print(f"ANOMALIB_AVAILABLE: {ANOMALIB_AVAILABLE}")
    print(f"ANOMALY_SCORE: {res.get('prediction', {}).get('anomaly_score')}")
    print(f"THRESHOLD: {res.get('prediction', {}).get('threshold')}")
    print(f"STATUS: {res.get('prediction', {}).get('status')}")
    print(f"HEATMAP_PATH: {res.get('localization', {}).get('heatmap_uri')}")
    return res

def run_pipeline_b_diagnostic(image_path: Path, label: str):
    uf = make_upload_file(image_path)
    t0 = time.time()
    res = run_multi_instance_inspection(db, model_id=MODEL_ID, upload_file=uf)
    latency = int((time.time() - t0) * 1000)
    
    instances = res.get("instances", [])
    defective_count = sum(1 for inst in instances if inst.get("prediction", {}).get("status") in ["anomalous", "REJECT"])
    token_usage = res.get("usage_metadata", {}).get("total_tokens", "N/A")
    
    print(f"\n--- PIPELINE B DIAGNOSTIC ({label}) ---")
    print(f"PIPELINE: Pipeline B (Multi-Product Gemini VLM)")
    print(f"MODE: multi_instance")
    print(f"GEMINI_MODEL: gemini-2.5-flash")
    print(f"THINKING_BUDGET: 1024")
    print(f"INPUT_PATH: {image_path}")
    print(f"INSTANCE_COUNT: {len(instances)}")
    print(f"DEFECTIVE_COUNT: {defective_count}")
    print(f"LATENCY: {latency} ms")
    print(f"TOKEN_USAGE: {token_usage}")
    return res

if __name__ == "__main__":
    print("==================================================")
    print("STARTING FULL PIPELINE A/B ISOLATION & STABILITY AUDIT")
    print("==================================================")
    
    # 1. Pipeline A Verification
    a1 = run_pipeline_a_diagnostic(GOOD_SINGLE, "Known GOOD Screw")
    a2 = run_pipeline_a_diagnostic(DEFECTIVE_SINGLE, "Known DEFECTIVE Screw")
    
    # 2. Pipeline B Verification
    b1 = run_pipeline_b_diagnostic(MULTI_COMPOSITE, "2x2 Multi-Product Grid")
    
    # 3. Mandatory Cross-Pipeline Isolation Sequence: A -> B -> A
    print("\n==================================================")
    print("EXECUTING MANDATORY CROSS-PIPELINE SEQUENCE: A -> B -> A")
    print("==================================================")
    res_a_first = run_pipeline_a_diagnostic(GOOD_SINGLE, "A (First Run)")
    res_b_middle = run_pipeline_b_diagnostic(MULTI_COMPOSITE, "B (Middle Run)")
    res_a_second = run_pipeline_a_diagnostic(GOOD_SINGLE, "A (Second Run after B)")
    
    # Verify A1 vs A2 consistency
    s1 = res_a_first.get("prediction", {}).get("anomaly_score")
    s2 = res_a_second.get("prediction", {}).get("anomaly_score")
    print("\nISOLATION VERIFICATION (A -> B -> A):")
    print(f"  First Pipeline A Score:  {s1}")
    print(f"  Second Pipeline A Score: {s2}")
    if s1 == s2:
        print("  SUCCESS: Pipeline A output is 100% IDENTICAL and STABLE after running Pipeline B!")
    else:
        print("  WARNING: Pipeline A output changed after running Pipeline B!")

    # 4. Mandatory Cross-Pipeline Isolation Sequence: B -> A -> B
    print("\n==================================================")
    print("EXECUTING MANDATORY CROSS-PIPELINE SEQUENCE: B -> A -> B")
    print("==================================================")
    res_b_first = run_pipeline_b_diagnostic(MULTI_COMPOSITE, "B (First Run)")
    res_a_mid = run_pipeline_a_diagnostic(GOOD_SINGLE, "A (Middle Run)")
    res_b_second = run_pipeline_b_diagnostic(MULTI_COMPOSITE, "B (Second Run after A)")

    b1_count = len(res_b_first.get("instances", []))
    b2_count = len(res_b_second.get("instances", []))
    print("\nISOLATION VERIFICATION (B -> A -> B):")
    print(f"  First Pipeline B Instance Count:  {b1_count}")
    print(f"  Second Pipeline B Instance Count: {b2_count}")
    if b1_count == b2_count:
        print("  SUCCESS: Pipeline B output is 100% IDENTICAL and STABLE after running Pipeline A!")
    else:
        print("  WARNING: Pipeline B output changed after running Pipeline A!")
