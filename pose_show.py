import cv2
import numpy as np
import camera

v0 = "data/record/video_0/1732262179527_0.mp4"
v1 = "data/record/video_0/1732262179527_1.mp4"

cap0 = camera.get_cv2_capture(v0)
cap1 = camera.get_cv2_capture(v1)


idx = 0
while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    # plot_name = f"pose/pose_3d_{idx}.png"
    # frame_plot = cv2.imread(plot_name)
    # frame_plot = cv2.resize(frame_plot, (1280, 720))

    # Check if frames were captured successfully
    if not ret1 or not ret0:
        print("Error: Could not read frames from one or both cameras.")
        break

    frame = np.hstack((frame0, frame0))
    cv2.imshow("YOLO11 Tracking", frame)

    idx +=1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


print(idx)
# 释放VideoCapture对象
cap0.release()
cap1.release()
cv2.destroyAllWindows()
