import numpy as np
import cv2
import time
import os

import utility as su

def single_picture_recording(cam_id,key="sample"):
    sample_folder = su.create_ouput_folder(key)

    print("single camera recording")
    capture = cv2.VideoCapture(cam_id)

    if not capture.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Could not read frame.")
            break


        cv2.imshow("0", frame)
        key = cv2.waitKey(1)
        if key == 32:
            timestampSTR = str(round(time.time() * 1000))

            img_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"saved Image:  {img_path}")

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("0")

def double_picture_recording(cam_id1, cam_id2, key="sample"):

    sample_folder = su.create_ouput_folder(key)
    print("double camera recording")
    capture0 = cv2.VideoCapture(cam_id1)
    capture1 = cv2.VideoCapture(cam_id2)

    if not capture0.isOpened() or not capture1.isOpened():
        print("Error: Could not open one or both cameras.")
        exit()

    while True:
        t0 = time.time()
        # Capture frames from both cameras
        ret0, frame0 = capture0.read()
        ret1, frame1 = capture1.read()

        # Check if frames were captured successfully
        if not ret0 or not ret1:
            print("Error: Could not read frames from one or both cameras.")
            break

        images = np.hstack((frame0, frame1))
        cv2.imshow("0", images)
        key = cv2.waitKey(1)

        if key == 32:
            timestampSTR = str(round(time.time() * 1000))
            # for stream0
            img_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id1}.jpg")
            cv2.imwrite(img_path, frame0)
            print(f"saved Image:  {img_path}")

            # for stream1
            img_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id2}.jpg")
            cv2.imwrite(img_path, frame1)
            print(f"saved Image:  {img_path}")

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("0")