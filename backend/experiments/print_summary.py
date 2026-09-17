import json

p = "c:/dev/Anomaly_Detector/backend/outputs/generic_patchcore/textile/failure_analysis/textile_failure_analysis.json"
data = json.load(open(p))

print("=== DISCOVERED FALSE POSITIVE IMAGES ===")
for fname in data["discovered_false_positives"]:
    res = next(r for r in data["all_results"] if r["filename"] == fname)
    print(f"  {fname:<16} | Raw Dist: {res['raw_anomaly_distance']:.4f} | Ratio: {res['calibrated_score']:.4f}")

print("\n=== DISCOVERED TRUE NEGATIVE IMAGES ===")
for fname in data["discovered_true_negatives"]:
    res = next(r for r in data["all_results"] if r["filename"] == fname)
    print(f"  {fname:<16} | Raw Dist: {res['raw_anomaly_distance']:.4f} | Ratio: {res['calibrated_score']:.4f}")

print("\n=== CONFUSION MATRIX & RATES ===")
print("  True Positives (TP) :", data["confusion_matrix"]["true_positives"])
print("  True Negatives (TN) :", data["confusion_matrix"]["true_negatives"])
print("  False Positives (FP):", data["confusion_matrix"]["false_positives"])
print("  False Negatives (FN):", data["confusion_matrix"]["false_negatives"])
print("  False Positive Rate :", f"{data['rates']['false_positive_rate_percent']}%")
print("  Recall / Sensitivity:", f"{data['rates']['recall_percent']}%")
print("  Overall Accuracy    :", f"{data['rates']['accuracy_percent']}%")
