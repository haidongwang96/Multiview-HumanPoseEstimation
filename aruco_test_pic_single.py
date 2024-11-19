from pyglet import image

import camera
import utility as su

import numpy as np
import cv2


import cv2
import numpy as np

def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, length):
    # 定义坐标轴在3D空间中的终点
    axis_points = np.float32([
        [0, 0, 0],            # 原点
        [length, 0, 0],       # X轴终点
        [0, length, 0],       # Y轴终点
        [0, 0, length],       # Z轴终点
    ]).reshape(-1, 3)

    # 将3D点投影到图像平面
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    print(imgpts)
    imgpts = imgpts.astype(int).reshape(-1, 2)

    # 获取原点坐标
    origin = tuple(imgpts[0])

    # 绘制X轴（红色）
    frame = cv2.line(frame, origin, tuple(imgpts[1]), (0, 0, 255), 3)
    # 绘制Y轴（绿色）
    frame = cv2.line(frame, origin, tuple(imgpts[2]), (0, 255, 0), 3)
    # 绘制Z轴（蓝色）
    frame = cv2.line(frame, origin, tuple(imgpts[3]), (255, 0, 0), 3)

    return frame




# 1280*720
mtx0 = np.array([[791.5837032, 0., 641.05413647],
                 [0., 791.94049493, 311.24557671],
                 [0., 0., 1.]])

intrin = camera.Intrinsics(mtx0)
dist1 = np.zeros(5)
dist0 = np.array([[1.56754836e-02, 8.64003826e-01, -7.33367826e-03, 1.41220244e-03, -6.44556463e+00]])


image_folder_path ="data/record/aruco_0"
images_names = su.collect_images_by_index(image_folder_path, 0)

images = [cv2.imread(imname, 1) for imname in images_names]
print(f"{len(images)} images found.")

frame0 = images[0]

r_vec,t_vec = camera.estimate_single_marker_r_t(frame0, mtx0, dist0)
T = camera.extrinsics(r_vec, t_vec)

# frame0 =cv2.drawFrameAxes(frame0, mtx0, dist0, r_vec, t_vec, 0.05)
# cv2.imwrite("a.jpg", frame0)


length = 0.05  # 坐标轴的长度
img_with_axes = draw_axes(frame0, mtx0, dist1, r_vec, t_vec, length)
cv2.imwrite("d.jpg", frame0)



# cv2计算
axis_points = np.float32([
        [0, 0, 0],            # 原点
        [length, 0, 0],       # X轴终点
        [0, length, 0],       # Y轴终点
        [0, 0, length],       # Z轴终点
    ]).reshape(-1, 3)
# cv2 映射需要r_vec
points, _ = cv2.projectPoints(axis_points, r_vec, t_vec, mtx0, dist0)
points = points.astype(int).reshape(-1, 2)
print(points)

frame0 = cv2.line(frame0, points[0], points[1], (0, 0, 255), 3)
cv2.imwrite("c.jpg", frame0)






