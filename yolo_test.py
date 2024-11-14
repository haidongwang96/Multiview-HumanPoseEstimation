import time

import cv2
import numpy as np

import utility as su
import model


fontScale = 0.5
thickness = 1


yolo = model.YoloV8("data/weights/yolov8n.pt")
yolo.model.conf = 0.8
colors = su.generate_distinct_colors(len(yolo.class_names.keys()))
print(yolo.class_names)
print(colors)


# Initialize the camera (use 0 for the default camera)
capture = cv2.VideoCapture(0)


if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    t0 = time.time()
    ret, frame = capture.read()


    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the resulting frame
    minibatch = [frame]
    preds = yolo.process([frame])



    for i, pred in enumerate(preds[0]):

        if pred is None: continue
        for region in pred:
            # print(region.toint().bbox)
            x1, y1, x2, y2 = region.toint().bbox
            label_idx = yolo.class_idx_by_name[region.label]
            # img_crop = img[y1:y2, x1:x2]
            cv2.putText(minibatch[i], f"{region.label} {region.score:.2f} ", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, colors[label_idx], thickness, cv2.LINE_AA)
            c_x, c_y = region.center
            cv2.circle(minibatch[i], (int(c_x), int(c_y)), 3, colors[label_idx], -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[label_idx], 2)

            # show FPS
            fps = 1 / (time.time() - t0)
            cv2.putText(minibatch[0], f"FPS: {fps:.1f} ", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 255, 0), thickness, cv2.LINE_AA)


    cv2.imshow('Camera Frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
capture.release()
cv2.destroyAllWindows()