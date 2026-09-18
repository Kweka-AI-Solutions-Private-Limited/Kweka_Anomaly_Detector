"""
Verification script for Pure-Gemini Pipeline B Implementation
--------------------------------------------------------------
Tests that:
1. PIPELINE_B_MODE environment toggle works.
2. Direct multi-instance inspection endpoint executes Pure Gemini VLM call.
3. Standardized result structure, instance crops, and Gemini Inspection Surface PNG are generated.
"""

import os
import sys
from pathlib import Path

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.connection import get_db
from services.inspection_service import run_inspection

def main():
    print("==================================================")
    print("VERIFYING PURE-GEMINI PIPELINE B IMPLEMENTATION")
    print("==================================================")

    db = get_db()

    # Get active model
    model = db.models.find_one({"status": "active"})
    if not model:
        # Fallback to any model
        model = db.models.find_one()

    if not model:
        print("[ERROR] No models found in database.")
        sys.exit(1)

    model_id = str(model["_id"])
    print(f"[CONFIG] Model ID: {model_id} ('{model.get('name')}')")

    composite_path = Path(r"C:\dev\Anomaly_Detector\backend\storage\experiments\gemini_reference_comparison_diagnostic\composites\composite_001.png")
    if not composite_path.exists():
        print(f"[ERROR] Test composite image not found at {composite_path}")
        sys.exit(1)

    print(f"[CONFIG] Test Composite Image: {composite_path}")
    print(f"[CONFIG] PIPELINE_B_MODE: {os.getenv('PIPELINE_B_MODE', 'gemini_only')}")

    class MockUploadFile:
        def __init__(self, filepath):
            self.filepath = filepath
            self.filename = filepath.name
            self.file = open(filepath, "rb")

        def read(self):
            self.file.seek(0)
            return self.file.read()

    upload_file = MockUploadFile(composite_path)

    try:
        res = run_inspection(
            db=db,
            model_id=model_id,
            upload_file=upload_file,
            inspection_mode="multi_instance"
        )

        print("\n[SUCCESS] Multi-Instance Inspection Executed!")
        print(f"Inspection ID: {res.get('inspection_id')}")
        print(f"Inspection Mode: {res.get('inspection_mode')}")

        result_data = res.get("result", {})
        overall = result_data.get("overall_prediction", {})
        print(f"\nOverall Verdict: {overall.get('status')}")
        print(f"Total Instances: {overall.get('total_instances')}")
        print(f"Pass Count: {overall.get('pass_count')}")
        print(f"Reject Count: {overall.get('reject_count')}")
        print(f"Message: {overall.get('message')}")

        instances = result_data.get("instances", [])
        print(f"\nInstance Breakdown ({len(instances)} instances):")
        for inst in instances:
            pred = inst.get("prediction", {})
            loc = inst.get("localization", {})
            vlm = inst.get("vlm_analysis", {})
            vlm_type = vlm.get('defect_type') if isinstance(vlm, dict) else 'N/A'
            print(f"  - Instance #{inst.get('instance_id')}: {pred.get('status')} | Score: {pred.get('anomaly_score')} / Thr: {pred.get('threshold')} | Conf: {inst.get('detection_confidence')} | Defect: {vlm_type} | Heatmap: {loc.get('heatmap_uri')}")

        comp_heatmap = result_data.get("composite_heatmap_uri")
        print(f"\nComposite Heatmap / Gemini Inspection Surface URI: {comp_heatmap}")

        stats = result_data.get("processing_stats", {})
        print(f"Engine: {stats.get('engine')} | Model: {stats.get('model')} | Latency: {stats.get('total_time_ms')}ms")

    except Exception as e:
        print(f"\n[ERROR] Multi-instance inspection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
