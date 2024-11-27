import numpy as np
import cv2
import scipy

import camera
import utility as su


class Extrinsic:

    def __init__(self, R, t):
        self.pose_mat = np.zeros((4, 4))
        # cv2.Rodrigues() 是 OpenCV 中用于旋转矩阵和旋转向量互相转换的重要函数
        if R.shape != (3, 3):
            R, _ = cv2.Rodrigues(R)

        self.pose_mat[:3, :3] = R
        self.pose_mat[:3, 3] = t.flatten()
        self.pose_mat[3, 3] = 1

    def __str__(self):
        return str(self.pose_mat)

    def transform(self, points_3d):
        # have not been tested yet
        if points_3d.shape[1] != 3:
            points_3d = points_3d.reshape(-1, 3)
            print("l")
        new_points_3d = (self.R() @ points_3d.T).T + self.t().T  # 变换
        return new_points_3d


    def R(self):
        # 旋转矩阵
        return self.pose_mat[:3, :3]

    def r_vec(self):
        # 旋转向量
        r_vec, _ = cv2.Rodrigues(self.R())
        return r_vec

    def t(self):
        return self.pose_mat[:3, 3]

    def R_inv(self):
        return self.R().T

    def t_inv(self):
        t_vec = self.t().reshape(3,1)
        return -self.R_inv() @ t_vec

    def inverse_transformation(self):
        return Extrinsic(self.R_inv(), self.t_inv())

    def save(self, path):
        out_c0_extrin = {}
        out_c0_extrin["R"] = self.R().tolist()
        out_c0_extrin["T"] = self.t().tolist()
        su.write_json_file(out_c0_extrin, path)

def extr_load(path):
    d = su.read_json_file(path)
    R = np.array(d["R"])
    t = np.array(d["T"])
    return Extrinsic(R, t)


class Intrinsics:

    def __init__(self, cmtx, dist=None):
        """
        cmtx =  [[     603.57           0      319.55]
                 [          0      603.16      242.55]
                 [          0           0           1]]
        """

        self.fx = cmtx[0][0]
        self.fy = cmtx[1][1]
        self.cx = cmtx[0][2]
        self.cy = cmtx[1][2]

        self.dist = dist

    def get_cam_mtx(self):
        m = [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
        return np.array(m)

def intr_load(path):
    d = su.read_json_file(path)
    cmtx = np.array(d['intrinsic'])
    dist = np.array(d['distortion_coefs'])

    return Intrinsics(cmtx, dist)


class ProjectionMatrix():

    def __init__(self, intr: Intrinsics, extr: Extrinsic):
        self.intr = intr
        self.extr = extr

    def get_projection_matrix(self):
        """
        3*3 @ 3*4
        """
        return self.intr.get_cam_mtx() @ self.extr.pose_mat[:3, :]

def get_self_transformation_extrinsic():
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))
    return Extrinsic(R0, T0)

def cam1_to_cam2_transformation(extr1: Extrinsic, extr2: Extrinsic):
    """
    已知 Cam1和 Cam2到marker坐标系的旋转平移矩阵，通过marker坐标系做桥梁，计算出CAM1 到CAM2的变换矩阵
    详细数学计算见 IDEA.md 'cam1_to_cam2_transformation'
    """
    R1_2 = extr2.R() @ extr1.R_inv()
    t1_2 = extr2.t() - R1_2 @ extr1.t()
    return Extrinsic(R1_2,t1_2)

def project_2d_withdepth_to_3d(points2d, depth, intr: Intrinsics):
    """
    将图像中2d的点，投射到3d空间中
    理论上，2d到3d只能变换出一条线，所以此方法需要确定一个depth，给定空间中的位置\
    详细数学计算见 IDEA.md 'project_2d_withdepth_to_3d'
    """
    pixel_x, pixel_y = points2d
    X = depth * (pixel_x - intr.cx) / intr.fx
    Y = depth * (pixel_y - intr.cy) / intr.fy
    return np.array([X, Y, depth])

def project_3d_to_2d(points_3d, extr:Extrinsic, intr:Intrinsics):
    """
    todo: 此函数包含两个部分，拆分！
    将marker坐标系下的3d点投射到相机坐标系下，外参为 marker->cam
    将3d点转换成2d点
    :param extr: 旋转+平移
    :param intr: 相机内参
    :return:
    """
    point_num = len(points_3d)
    R = extr.R()
    tvec = extr.t().reshape(3,1)

    P_cam = (R @ points_3d.T).T + tvec.T # 变换
    P_cam = P_cam.reshape(point_num,3)
    x = P_cam[:, 0] / P_cam[:, 2]
    y = P_cam[:, 1] / P_cam[:, 2]

    u = intr.fx * x + intr.cx
    v = intr.fy * y + intr.cy
    img_points = np.vstack((u, v)).T
    return img_points.astype(int)

def triangulation(P0, P1, pts0, pts1):

    def DLT(P1, P2, point1, point2):
        A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))
        B = A.transpose() @ A

        U, s, Vh = scipy.linalg.svd(B, full_matrices=False)

        return Vh[3, 0:3] / Vh[3, 3]

    p3ds = []
    for uv0, uv1 in zip(pts0, pts1):
        _p3d = DLT(P0, P1, uv0, uv1)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
    return p3ds


def check_duo_calibration(extr0: Extrinsic, intr0: Intrinsics, extr1: Extrinsic, intr1: Intrinsics, _zshift =0.5 ):
    """
    这个函数测试了cam1_to_cam2_transformation的正确性
    注意：由于cap0和cap1，由于读取的顺序的问题可能是反的，所以显示的时候可能会有问题
    :param extr0: 本身相机坐标，无任何变换的外参
    :param extr1: 由cam2变换到cam1的外参
    """


    # define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = su.get_axis_points(length=0.05)
    z_shift = np.array([0., 0., _zshift]).reshape((1, 3))
    draw_axes_points = coordinate_points + z_shift

    #使用opencv方法
    imgpts0 = camera.project_3d_to_2d(draw_axes_points, extr0, intr0)
    imgpts1 = camera.project_3d_to_2d(draw_axes_points, extr1, intr1)

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

        frame0 = su.draw_axes(frame0, imgpts1)
        frame1 = su.draw_axes(frame1, imgpts0)
        frame = np.hstack((frame0, frame1))
        cv2.imshow('frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放VideoCapture对象
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()



