import cv2
import numpy as np
from pathlib import Path

data_dir = Path("c:/dev/Anomaly_Detector/backend/data/dtd")
images = list(data_dir.glob("*.png"))

for img_path in sorted(images)[:10]:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold screw foreground
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(c)
        print(f"Image {img_path.name}: Screw bounding box = x:{bx}, y:{by}, w:{bw}, h:{bh} in {w}x{h} image")
