import numpy as np

import camera
import cv2
import utility as su






mtx0 = np.array([[791.5837032, 0., 641.05413647],
                 [0., 791.94049493, 311.24557671],
                 [0., 0., 1.]])
dist0 = np.array([[1.56754836e-02, 8.64003826e-01, -7.33367826e-03, 1.41220244e-03, -6.44556463e+00]])


image_folder_path ="data/record/aruco_0"
images_names = su.collect_images_by_index(image_folder_path, 0)

images = [cv2.imread(imname, 1) for imname in images_names]
print(f"{len(images)} images found.")
frame0 = images[0]
rvec,tvec = camera.estimate_single_marker_r_t(frame0, mtx0, dist0)

length = 0.05  # 坐标轴的长度
axis_points = np.float32([
    [0, 0, 0],  # 原点
    [length, 0, 0],  # X轴终点
    [0, length, 0],  # Y轴终点
    [0, 0, length],  # Z轴终点
]).reshape(-1, 3)

extr0 = camera.Extrinsic(rvec, tvec)
intr0 = camera.Intrinsics(mtx0)
imgpts = camera.project_3d_to_2d(axis_points, extr0, intr0)
#print(imgpts)

img_with_axes = su.draw_axes(frame0, imgpts)
cv2.imwrite("a.jpg", frame0)


