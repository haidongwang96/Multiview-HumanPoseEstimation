import numpy as np
import cv2
import cv2.aruco as aruco
import math

import camera

parameters = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
detector = aruco.ArucoDetector(aruco_dict, parameters)
font = cv2.FONT_HERSHEY_SIMPLEX


def estimate_pose(frame, mtx, dist):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测图像中的ArUco标记
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    # 如果检测到ArUco标记
    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):  # 遍历每个检测到的ArUco标记
            # 估计ArUco标记的姿态（旋转向量和平移向量）
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.20, mtx, dist)
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.05)

        aruco.drawDetectedMarkers(frame, corners)

    # 如果没有检测到ArUco标记，则在图像上显示"No Ids"
    else:
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame  # 返回处理后的图像

def estimate_single_marker_r_t(frame, mtx, dist=np.array([0,0,0,0,0]),  marker_length=0.2):
    """
    distortion在这里统一忽略 直接带入[0,0,0,0,0]
    """
    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测图像中的ArUco标记
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    # 如果检测到ArUco标记
    if ids is not None and len(ids) == 1:
        # 估计ArUco标记的姿态（旋转向量和平移向量）
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], marker_length, mtx, dist)
    else:
        print("No Ids")

    return rvec, tvec




