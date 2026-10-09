import json
from pathlib import Path

p_json = Path("c:/dev/Anomaly_Detector/backend/outputs/generic_patchcore/textile/textile_calibration_diagnostic.json")
data = json.loads(p_json.read_text())

s_80 = data["calibration_80_20_split"]["statistics"]
f_80 = data["calibration_80_20_split"]["known_good_test_fpr"]

s_loo = data["calibration_leave_one_out"]["statistics"]
f_loo = data["calibration_leave_one_out"]["known_good_test_fpr"]

kg = data["known_good_test_raw_distances"]
stability_msg = data["threshold_stability_assessment"]

md = f"""# Textile PatchCore Calibration Diagnostic Report

## 1. Out-of-Sample GOOD Calibration Distances (N=21 Reference Images)

### A. Split-Sample 80/20 Calibration (N=16 Memory Ref, N=5 Calibration Samples)
- **Individual Calibration Distance Values**:
"""

for k, v in s_80["individual_values"].items():
    md += f"  - `{k}`: `{v:.4f}`\n"

md += f"""- **Mean**: `{s_80['mean']:.4f}`
- **Std Dev**: `{s_80['std']:.4f}`
- **90th Percentile Threshold**: `{s_80['p90']:.4f}`
- **95th Percentile Threshold**: `{s_80['p95']:.4f}`
- **97.5th Percentile Threshold**: `{s_80['p97_5']:.4f}`
- **99th Percentile Threshold**: `{s_80['p99']:.4f}`

### B. Leave-One-Out (LOO) Calibration (All N=21 Out-of-Sample Scores)
- **Individual Calibration Distance Values**:
"""

for k, v in s_loo["individual_values"].items():
    md += f"  - `{k}`: `{v:.4f}`\n"

md += f"""- **Mean**: `{s_loo['mean']:.4f}`
- **Std Dev**: `{s_loo['std']:.4f}`
- **90th Percentile Threshold**: `{s_loo['p90']:.4f}`
- **95th Percentile Threshold**: `{s_loo['p95']:.4f}`
- **97.5th Percentile Threshold**: `{s_loo['p97_5']:.4f}`
- **99th Percentile Threshold**: `{s_loo['p99']:.4f}`

---

## 2. Raw Anomaly Distances for 10 Known GOOD Test Images

Full Memory Bank built on ALL 21 GOOD reference images:
"""

for k, v in kg.items():
    md += f"- `{k}`: **`{v:.4f}`**\n"

md += f"""
---

## 3. False Positive Rate (FPR) Analysis on 10 Known GOOD Test Images

### A. Split-Sample 80/20 Calibration (N=5 Calibration Samples)
| Percentile Level | Threshold Value | Classified Anomalous (/10) | FPR (%) | Images Exceeding Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **90th Percentile** | `{f_80['p90']['threshold']:.4f}` | `{f_80['p90']['exceed_count']}` | `{f_80['p90']['fpr_percent']:.1f}%` | `{", ".join(f_80['p90']['exceed_images'])}` |
| **95th Percentile** | `{f_80['p95']['threshold']:.4f}` | `{f_80['p95']['exceed_count']}` | `{f_80['p95']['fpr_percent']:.1f}%` | `{", ".join(f_80['p95']['exceed_images'])}` |
| **97.5th Percentile** | `{f_80['p97_5']['threshold']:.4f}` | `{f_80['p97_5']['exceed_count']}` | `{f_80['p97_5']['fpr_percent']:.1f}%` | `{", ".join(f_80['p97_5']['exceed_images'])}` |
| **99th Percentile** | `{f_80['p99']['threshold']:.4f}` | `{f_80['p99']['exceed_count']}` | `{f_80['p99']['fpr_percent']:.1f}%` | `{", ".join(f_80['p99']['exceed_images'])}` |

### B. Leave-One-Out (LOO) Calibration (N=21 Out-of-Sample Validation Scores)
| Percentile Level | Threshold Value | Classified Anomalous (/10) | FPR (%) | Images Exceeding Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **90th Percentile** | `{f_loo['p90']['threshold']:.4f}` | `{f_loo['p90']['exceed_count']}` | `{f_loo['p90']['fpr_percent']:.1f}%` | `{", ".join(f_loo['p90']['exceed_images'])}` |
| **95th Percentile** | `{f_loo['p95']['threshold']:.4f}` | `{f_loo['p95']['exceed_count']}` | `{f_loo['p95']['fpr_percent']:.1f}%` | `{", ".join(f_loo['p95']['exceed_images'])}` |
| **97.5th Percentile** | `{f_loo['p97_5']['threshold']:.4f}` | `{f_loo['p97_5']['exceed_count']}` | `{f_loo['p97_5']['fpr_percent']:.1f}%` | `{", ".join(f_loo['p97_5']['exceed_images'])}` |
| **99th Percentile** | `{f_loo['p99']['threshold']:.4f}` | `{f_loo['p99']['exceed_count']}` | `{f_loo['p99']['fpr_percent']:.1f}%` | `{", ".join(f_loo['p99']['exceed_images'])}` |

---

## 4. Assessment of Threshold Instability (N=21 Reference Dataset)

{stability_msg}
"""

p_md = Path("c:/dev/Anomaly_Detector/backend/outputs/generic_patchcore/textile/textile_calibration_diagnostic.md")
p_md.write_text(md)
print("Markdown report written successfully to:", p_md)
