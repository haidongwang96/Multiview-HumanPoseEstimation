import camera
import utility as su
import numpy as np
import cv2



# 1280*720
# mtx0 = np.array([[791.5837032, 0., 641.05413647],
#                  [0., 791.94049493, 311.24557671],
#                  [0., 0., 1.]])
# dist0 = np.array([[1.56754836e-02, 8.64003826e-01, -7.33367826e-03, 1.41220244e-03, -6.44556463e+00]])
#
# mtx1 = np.array([[781.18543401, 0., 652.45577192],
#                  [0., 780.81321374, 338.19187737],
#                  [0., 0., 1.]])
# dist1 = np.array([[0.05427265,  0.00726026,  0.00728199, -0.00566807, -0.22047885]])

mtx0, dist0 = camera.load_intrinsic_calibration_parameters("data/camera_parameters/camera_0_intrinsics.json")
mtx1, dist1 = camera.load_intrinsic_calibration_parameters("data/camera_parameters/camera_1_intrinsics.json")


cap0 = camera.get_cv2_capture(0)
cap1 = camera.get_cv2_capture(1)

if not cap0.isOpened() or not cap1.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    # Check if frames were captured successfully
    if not ret1 or not ret0:
        print("Error: Could not read frames from one or both cameras.")
        break

    # 调用estimate_pose函数对当前帧进行姿态估计和标记检测
    frame0 = camera.estimate_pose(frame0, mtx0, dist0)
    frame1 = camera.estimate_pose(frame1, mtx1, dist1)

    frame = np.hstack((frame0, frame1))
    cv2.imshow('frame', frame)

    # 等待按键输入，如果按下键盘上的'q'键则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放VideoCapture对象c
cap0.release()
cap1.release()
cv2.destroyAllWindows()