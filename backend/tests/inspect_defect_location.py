import cv2
import numpy as np
from pathlib import Path

img_path = Path("c:/dev/Anomaly_Detector/backend/data/dtd/000_heatmap.png")
img_bgr = cv2.imread(str(img_path))

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# Compute localized intensity difference / gradient variance
blur = cv2.GaussianBlur(gray, (5, 5), 0)
diff = cv2.absdiff(gray, blur)

# Find peak location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(diff)
print(f"Peak intensity variance location on {img_path.name}: {max_loc}, Max Val: {max_val}")

# Also analyze edge density variance across 4 quadrants
h, w = gray.shape
q1 = np.std(gray[:h//2, :w//2])
q2 = np.std(gray[:h//2, w//2:])
q3 = np.std(gray[h//2:, :w//2])
q4 = np.std(gray[h//2:, w//2:])
print(f"Quadrant standard deviations: Q1(top-left)={q1:.2f}, Q2(top-right)={q2:.2f}, Q3(bottom-left)={q3:.2f}, Q4(bottom-right)={q4:.2f}")
