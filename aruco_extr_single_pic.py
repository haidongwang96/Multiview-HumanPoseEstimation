import camera
import utility as su
import numpy as np
import cv2
import glob

"""
双相机aruco外参标定
"""

SAVE = False

cam_ids=[2,4]

# 读取已标定内参
mtx0, dist0 = camera.load_intrinsic_calibration_parameters(f"data/camera_parameters/camera_{cam_ids[0]}_intrinsics.json")
mtx1, dist1 = camera.load_intrinsic_calibration_parameters(f"data/camera_parameters/camera_{cam_ids[1]}_intrinsics.json")

# 加载aruco图像对
aruco_dir = "data/record_ubuntu/landmark_0"
idx = 3

cam0_path = glob.glob(f"{aruco_dir}/*_{cam_ids[0]}.jpg")[idx]
cam1_path = glob.glob(f"{aruco_dir}/*_{cam_ids[1]}.jpg")[idx]
frame0 = cv2.imread(cam0_path)
frame1 = cv2.imread(cam1_path)

# 依次校准，分别得到相机的旋转矩阵，平移向量（世界坐标系 -> 相机坐标系)
rvec0, tvec0 = camera.estimate_single_marker_r_t(frame0, mtx0)
rvec1, tvec1 = camera.estimate_single_marker_r_t(frame1, mtx1)

# 将aruco显示到frame上
frame0_marked = camera.estimate_pose(frame0, mtx0, dist0)
frame1_marked = camera.estimate_pose(frame1, mtx1, dist1)

# 显示效果
frame = np.hstack((frame0_marked, frame1_marked))
cv2.imshow('frame', frame)
cv2.waitKey(0)  # 0 means wait indefinitely
cv2.destroyAllWindows()

if SAVE:
    # 分别保存外参
    extr0 = camera.Extrinsic(rvec0, tvec0)
    path_Extr_cw_c0 = f"data/camera_parameters/Extr_C_world_C_{cam_ids[0]}.json"
    extr0.save(path_Extr_cw_c0)
    extr1 = camera.Extrinsic(rvec1, tvec1)
    path_Extr_cw_c1 = f"data/camera_parameters/Extr_C_world_C_{cam_ids[1]}.json"
    extr1.save(path_Extr_cw_c1)




