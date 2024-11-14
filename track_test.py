import cv2
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from collections import defaultdict

import time
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("data/weights/yolo11n.pt")

# Open the video file
# video_path = "WIN_20241113_16_34_36_Pro.mp4"
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        t0 = time.time()
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, half=True, conf=0.5, iou=0.7, classes=[0], verbose=False, tracker="bytetrack.yaml")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        fps = 1 / (time.time() - t0)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f} ", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

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

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()