import cv2
import os
from tqdm import tqdm

# change paths here
ALPHA_VIDEO_PATH = 'rvm_alpha.mp4' 
OUT_DIR = 'dataset/name/alpha'

os.makedirs(OUT_DIR, exist_ok=True)

# open video
cap = cv2.VideoCapture(ALPHA_VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"🔄 Rotating and extracting {frame_count} alpha frames...")

for i in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break

    # rotate counterclockwise (90 degrees)
    # comment this out if needed. it was necessary for the test video used
    rotated_alpha = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # grays cale masking
    gray_alpha = cv2.cvtColor(rotated_alpha, cv2.COLOR_BGR2GRAY)

    # naming convention
    frame_name = f"{i:05d}.png"
    cv2.imwrite(os.path.join(OUT_DIR, frame_name), gray_alpha)

cap.release()
print(f"✅ Alpha frames saved to: {OUT_DIR}")
