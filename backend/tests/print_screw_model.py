from bson import ObjectId
from db.connection import get_db

db = get_db()
m = db.models.find_one({"_id": ObjectId("6aa3aa1d4067031f17c51123")})
print("MODEL:", m)

versions = list(db.model_versions.find({"model_id": ObjectId("6aa3aa1d4067031f17c51123")}))
for v in versions:
    print(f"\nVERSION #{v.get('version_number')} ID: {v['_id']}")
    print("  Status:", v.get("status"))
    print("  Calibration:", v.get("calibration"))
    print("  Artifacts:", v.get("artifacts"))
