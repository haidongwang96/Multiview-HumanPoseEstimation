import cv2
import time

import numpy as np

import utility as su
import model


fontScale = 0.5
thickness = 1

yolo = model.YoloV8("weights/yolov8n.pt")
yolo.model.conf = 0.8
colors = su.generate_distinct_colors(len(yolo.class_names.keys()))
print(yolo.class_names)
print(colors)
# Initialize both cameras
capture1 = cv2.VideoCapture(0)
capture2 = cv2.VideoCapture(1)


# Set the resolution for both cameras
# capture1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# capture1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# capture2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# capture2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Set video codec and create VideoWriter objects for both cameras
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out1 = cv2.VideoWriter('camera1_output.mp4', fourcc, 20.0, (int(capture1.get(3)), int(capture1.get(4))))
# out2 = cv2.VideoWriter('camera2_output.mp4', fourcc, 20.0, (int(capture2.get(3)), int(capture2.get(4))))

if not capture1.isOpened() or not capture2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

recording = False

while True:
    t0 = time.time()
    # Capture frames from both cameras
    ret1, frame1 = capture1.read()
    ret2, frame2 = capture2.read()

    # Check if frames were captured successfully
    if not ret1 or not ret2:
        print("Error: Could not read frames from one or both cameras.")
        break

    minibatch = [frame1,frame2]
    preds = yolo.process(minibatch)[0]

    for i, pred in enumerate(preds):

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
            cv2.rectangle(minibatch[i], (x1, y1), (x2, y2), colors[label_idx], 2)

            # show FPS
            fps = 1 / (time.time() - t0)
            cv2.putText(minibatch[0], f"FPS: {fps:.1f} ", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

    # Display the frames from both cameras
    cv2.imshow('Camera 1 Frame', frame1)
    cv2.imshow('Camera 2 Frame', frame2)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF



# Release the captures and output files, and close all windows
capture1.release()
capture2.release()
cv2.destroyAllWindows()
