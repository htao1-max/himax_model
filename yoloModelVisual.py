"""
YOLOv8n Video Processor
=======================
Processes an MP4 file using a YOLOv8n model with 2 classes.
Draws bounding boxes and labels on each frame and saves the annotated video.

Requirements:
    pip install ultralytics opencv-python
"""

import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── Configuration (edit these) ───────────────────────────────
SOURCE      = "ezgif.com-split.mp4"       # Input video path
MODEL_PATH  = "customModel.pt"         # YOLOv8n weights path
OUTPUT      = "output1.mp4"      # Output video path
CONF_THRESH = 0.25              # Confidence threshold
IOU_THRESH  = 0.45              # IoU threshold for NMS
SHOW        = False             # Show live preview window

# ── Load model ───────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

class_names = model.names  # e.g. {0: 'classA', 1: 'classB'}
print(f"Classes: {class_names}")

# ── Open source video ────────────────────────────────────────
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    sys.exit(f"Error: cannot open video '{SOURCE}'")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Source  : {SOURCE}")
print(f"Size    : {width}x{height} @ {fps:.1f} FPS")
print(f"Frames  : {total_frames}")

# ── Set up output writer ─────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (width, height))

# ── Colours per class (BGR) ──────────────────────────────────
colors = {
    0: (0, 255, 0),   # green  – class 0
    1: (0, 120, 255),  # orange – class 1
}

# ── Frame loop ───────────────────────────────────────────────
frame_idx = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(
        frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False,
    )

    # Draw detections
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = colors.get(cls_id, (255, 255, 255))
            label = f"{class_names[cls_id]} {conf:.2f}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

    writer.write(frame)
    frame_idx += 1

    # Progress
    if frame_idx % 100 == 0 or frame_idx == total_frames:
        elapsed = time.time() - start
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"  [{frame_idx}/{total_frames}] {proc_fps:.1f} FPS")

    # Optional live preview
    if SHOW:
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview closed by user.")
            break

# ── Cleanup ──────────────────────────────────────────────────
cap.release()
writer.release()
if SHOW:
    cv2.destroyAllWindows()

elapsed = time.time() - start
print(f"\nDone – {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} FPS)")
print(f"Saved to: {Path(OUTPUT).resolve()}")