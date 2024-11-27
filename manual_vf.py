import cv2
import glob
import numpy as np

import utility as su

def is_point_in_triangle(A, B, C, P):
    def vector_cross_product(Z, X, Y):
        return (Y[0] - Z[0]) * (X[1] - Z[1]) - (X[0] - Z[0]) * (Y[1] - Z[1])
    cross1 = vector_cross_product(A, B, P)
    cross2 = vector_cross_product(B, C, P)
    cross3 = vector_cross_product(C, A, P)

    # 检查叉积的符号是否一致
    if ((cross1 >= 0 and cross2 >= 0 and cross3 >= 0) or
        (cross1 <= 0 and cross2 <= 0 and cross3 <= 0)):
        return True
    else:
        return False

def is_point_in_quadrilateral(landmarks, P):
    A, B, C, D = landmarks
    # 检查点是否在三角形ABC内部
    if is_point_in_triangle(A, B, C, P):
        return True
    # 检查点是否在三角形ACD内部
    elif is_point_in_triangle(A, C, D, P):
        return True
    else:
        return False

# # 示例坐标
# A = (0, 0)
# B = (4, 0)
# C = (4, 3)
# D = (0, 3)
# P = (2, 1)
#
# if is_point_in_quadrilateral(A, B, C, D, P):
#     print("点P在四边形内部")
# else:
#     print("点P在四边形外部")


cam_2_ldms = su.read_list_file("data/annotation/mouse_click/landmark_0/1732607373680_2.txt", " ")
cam_4_ldms = su.read_list_file("data/annotation/mouse_click/landmark_0/1732607373680_4.txt", " ")

cam_2_ldms = np.array(cam_2_ldms, dtype=int)
cam_4_ldms = np.array(cam_4_ldms, dtype=int)

p2 = "data/record_ubuntu/landmark_0/1732607373680_2.jpg"
p4 = "data/record_ubuntu/landmark_0/1732607373680_4.jpg"


#
# cam_2_ldms = [[867, 610],[1005, 504],[789, 399],[639, 464]]
# cam_4_ldms = [[424, 512],[325, 380],[10, 482],[18, 691]]

p0 = [[816, 483],[928, 608]]
p1 = [[210, 497],[451,462]]

img2 = cv2.imread(p2)
su.draw_fence(img2, cam_2_ldms, color=(0, 255, 0), thickness=2)

img4 = cv2.imread(p4)
su.draw_fence(img4, cam_4_ldms, color=(0, 255, 0), thickness=2)

p_idx = 0

cv2.circle(img2, p0[p_idx], 2, (0, 0, 255), 2)
cv2.circle(img4, p1[p_idx], 2, (0, 0, 255), 2)

print(is_point_in_quadrilateral(cam_2_ldms, p0[p_idx]))
print(is_point_in_quadrilateral(cam_4_ldms, p1[p_idx]))

cv2.imshow("img2", img2)
cv2.imshow("img4", img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
