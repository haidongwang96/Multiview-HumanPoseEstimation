import cv2
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from collections import defaultdict

import time
import camera
from ultralytics import YOLO


def track(model, frame, track_history):
    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, half=True, conf=0.5, iou=0.7, classes=[0], verbose=False,
                          tracker="bytetrack.yaml")

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    if results is not None:
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
    return annotated_frame


# Load the YOLO11 model
model = YOLO("data/weights/yolo11n-pose.pt")

v0 = "data/record/video_0/1732262179527_0.mp4"
v1 = "data/record/video_0/1732262179527_1.mp4"

cap0 = camera.get_cv2_capture(v0)
cap1 = camera.get_cv2_capture(v1)

if not cap0.isOpened() or not cap1.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

track_history0 = defaultdict(lambda: [])
track_history1 = defaultdict(lambda: [])

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    # Check if frames were captured successfully
    if not ret1 or not ret0:
        print("Error: Could not read frames from one or both cameras.")
        break

    t0 = time.time()

    annotated_frame0 = track(model, frame0, track_history0)
    annotated_frame1 = track(model, frame1, track_history1)

    frame = np.hstack((annotated_frame0, annotated_frame1))
    fps = 1 / (time.time() - t0)
    cv2.putText(frame, f"FPS: {fps:.1f} ", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Display the annotated frame
    cv2.imshow("YOLO11 Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# 释放VideoCapture对象
cap0.release()
cap1.release()
cv2.destroyAllWindows()