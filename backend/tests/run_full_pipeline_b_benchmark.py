import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.connection import get_db
from services.inspection_service import run_inspection

class MockUploadFile:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = filepath.name
        self.file = open(filepath, "rb")

    def read(self):
        self.file.seek(0)
        return self.file.read()

def run_benchmark():
    print("===============================================================")
    print("PIPELINE B — BENCHMARK & METRICS EVALUATION SUITE")
    print("===============================================================")
    
    db = get_db()
    model = db.models.find_one({"status": "active"}) or db.models.find_one()
    if not model:
        print("[ERROR] No active model found in DB.")
        sys.exit(1)
        
    model_id = str(model["_id"])
    
    composites_dir = Path(r"C:\dev\Anomaly_Detector\backend\storage\experiments\gemini_reference_comparison_diagnostic\composites")
    composite_files = sorted(list(composites_dir.glob("composite_*.png")))
    
    if not composite_files:
        print(f"[ERROR] No composite images found in {composites_dir}")
        sys.exit(1)
        
    print(f"Found {len(composite_files)} composite test images.")
    
    results_summary = []
    
    total_instances_detected = 0
    total_pass = 0
    total_reject = 0
    total_latency_sec = 0.0
    
    for comp_path in composite_files:
        print(f"\n---------------------------------------------------------------")
        print(f"Testing {comp_path.name}...")
        upload_file = MockUploadFile(comp_path)
        
        t0 = time.time()
        res = run_inspection(
            db=db,
            model_id=model_id,
            upload_file=upload_file,
            inspection_mode="multi_instance"
        )
        t1 = time.time()
        duration = t1 - t0
        total_latency_sec += duration
        
        result_data = res.get("result", {})
        overall = result_data.get("overall_prediction", {})
        instances = result_data.get("instances", [])
        
        p_cnt = overall.get("pass_count", 0)
        r_cnt = overall.get("reject_count", 0)
        tot_cnt = overall.get("total_instances", len(instances))
        
        total_instances_detected += tot_cnt
        total_pass += p_cnt
        total_reject += r_cnt
        
        # Check polygon outputs
        has_prod_poly = any("product_polygon" in inst and len(inst.get("product_polygon", [])) >= 3 for inst in instances)
        has_def_poly = any("defect_polygon" in inst and len(inst.get("defect_polygon", [])) >= 3 for inst in instances if inst.get("prediction", {}).get("status") == "REJECT")
        
        print(f"Status: {overall.get('status')} | Total: {tot_cnt} | Pass: {p_cnt} | Reject: {r_cnt} | Latency: {duration:.2f}s")
        print(f"Product Polygon Present: {has_prod_poly} | Defect Polygon Present: {has_def_poly}")
        
        results_summary.append({
            "filename": comp_path.name,
            "status": overall.get('status'),
            "total_instances": tot_cnt,
            "pass_count": p_cnt,
            "reject_count": r_cnt,
            "latency": duration,
            "instances": instances
        })
        
    avg_latency = total_latency_sec / len(composite_files) if composite_files else 0
    print("\n===============================================================")
    print("BENCHMARK SUMMARY RESULTS")
    print("===============================================================")
    print(f"Total Images Benchmarked: {len(composite_files)}")
    print(f"Total Instances Detected Across Suite: {total_instances_detected}")
    print(f"Total Passed: {total_pass} | Total Rejected: {total_reject}")
    print(f"Average Latency: {avg_latency:.2f} seconds / inspection")
    print("===============================================================")

if __name__ == "__main__":
    run_benchmark()
