import sys, json
sys.path.insert(0, 'c:/dev/Anomaly_Detector/backend/src')
from db.connection import get_db

db = get_db()
docs = list(db.inspection_results.find().sort('_id', -1).limit(5))
docs.reverse()

lines = []
for i, d in enumerate(docs):
    instances = d.get('instances', [])
    status = d.get('overall_prediction', {}).get('status')
    lines.append(f"=================== Test Image #{i+1} ===================")
    lines.append(f"Overall Status: {status} | Total Instances: {len(instances)}")
    for inst in instances:
        iid = inst.get('instance_id')
        p_stat = inst.get('prediction', {}).get('status')
        vlm = inst.get('vlm_analysis', {})
        dtype = vlm.get('defect_type')
        sev = vlm.get('severity')
        exp = vlm.get('explanation')
        bbox = inst.get('bbox')
        lines.append(f"  Instance #{iid}: Status={p_stat}, DefectType={dtype}, Severity={sev}")
        lines.append(f"    Bbox={bbox}")
        lines.append(f"    Evidence: {exp}")

with open('tests/audit_output.txt', 'w') as f:
    f.write('\n'.join(lines))
print("Wrote audit_output.txt successfully!")
