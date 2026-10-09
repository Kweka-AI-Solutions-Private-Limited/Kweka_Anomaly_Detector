import cv2
import numpy as np
from pathlib import Path

for fname in ["000.png", "001.png", "004.png"]:
    p = Path("c:/dev/Anomaly_Detector/backend/data/dtd") / fname
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    
    # Compute high frequency edge intensity across screw length
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = np.abs(lap)
    
    # Split into 3 sections along diagonal (Head, Body, Threads/Tip)
    head = abs_lap[:int(h*0.35), :int(w*0.35)].mean()
    body = abs_lap[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)].mean()
    threads = abs_lap[int(h*0.65):, int(w*0.65):].mean()
    
    print(f"Image {fname}: Head laplacian={head:.2f}, Body laplacian={body:.2f}, Threads laplacian={threads:.2f}")
