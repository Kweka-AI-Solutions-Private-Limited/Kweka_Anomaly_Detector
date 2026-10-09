import sys
import subprocess

out = subprocess.check_output(['c:/dev/Anomaly_Detector/backend/venv/Scripts/python.exe', 'scripts/final_pipeline_a_verification.py'], text=True)
with open("final_report_output.txt", "w") as f:
    f.write(out)
print("Saved final_report_output.txt")
