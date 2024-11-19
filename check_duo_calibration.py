import cv2
import numpy as np

import camera
import utility as su

"""
此函数确认了双目相机 是否统一了坐标系
"""

mtx0 = np.array([[791.5837032, 0., 641.05413647],
                 [0., 791.94049493, 311.24557671],
                 [0., 0., 1.]])

mtx1 = np.array([[781.18543401, 0., 652.45577192],
                 [0., 780.81321374, 338.19187737],
                 [0., 0., 1.]])

image_folder_path ="data/record/aruco_0"

frame0 = cv2.imread(f"{image_folder_path}/1731723575664_0.jpg")
rvec0, tvec0 = camera.estimate_single_marker_r_t(frame0, mtx0)
frame1 = cv2.imread(f"{image_folder_path}/1731723575664_1.jpg")
rvec1, tvec1 = camera.estimate_single_marker_r_t(frame1, mtx1)


intr0 = camera.Intrinsics(mtx0)
intr1 = camera.Intrinsics(mtx1)

extr0 = camera.Extrinsic(rvec0, tvec0)
extr1 = camera.Extrinsic(rvec1, tvec1)

extr_cam1_to_cam0 = camera.cam1_to_cam2_transformation(extr1,extr0)
extr0_self = camera.get_self_transformation_extrinsic()

camera.check_duo_calibration(extr0_self, intr0, extr_cam1_to_cam0, intr1,_zshift=0.8)
