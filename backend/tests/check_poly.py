import sys, json
sys.path.insert(0, 'c:/dev/Anomaly_Detector/backend/src')
from db.connection import get_db

db = get_db()
docs = list(db.inspection_results.find().sort('_id', -1).limit(5))
docs.reverse()
for i, d in enumerate(docs):
    instances = d.get('instances', [])
    has_prod = any(len(inst.get('product_polygon') or []) >= 3 for inst in instances)
    has_def = any(len(inst.get('defect_polygon') or []) >= 3 for inst in instances)
    status = d.get('overall_prediction', {}).get('status')
    print(f"Test #{i+1}: Total={len(instances)}, Status={status}, HasProductPoly={has_prod}, HasDefectPoly={has_def}")
