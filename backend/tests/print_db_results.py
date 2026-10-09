import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.connection import get_db

db = get_db()
docs = list(db.inspection_results.find().sort('_id', -1).limit(5))
docs.reverse()

for d in docs:
    overall = d.get('overall_prediction', {})
    stats = d.get('processing_stats', {})
    print("\n----------------------------------------------------------")
    print(f"Inspection ID: {d.get('inspection_id')}")
    print(f"Engine: {stats.get('engine')} | Model: {stats.get('model')}")
    print(f"Overall Status: {overall.get('status')}")
    print(f"Total Instances: {overall.get('total_instances')}")
    print(f"Pass Count: {overall.get('pass_count')}")
    print(f"Reject Count: {overall.get('reject_count')}")
    instances = d.get('instances', [])
    for inst in instances:
        iid = inst.get('instance_id')
        pred = inst.get('prediction', {})
        vlm = inst.get('vlm_analysis', {})
        has_pp = len(inst.get('product_polygon', [])) >= 3
        has_dp = len(inst.get('defect_polygon', [])) >= 3
        print(f"  Instance #{iid}: Status={pred.get('status')}, Defect={vlm.get('defect_type')}, Severity={vlm.get('severity')}, ProdPoly={has_pp}, DefPoly={has_dp}")

