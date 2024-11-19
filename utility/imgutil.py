import cv2
import utility as su
import numpy as np

fontScale = 0.5
thickness = 1

def get_axis_points(length=0.05):
    coordinate_points = np.float32([[0, 0, 0],  # 原点
                                    [length, 0, 0],  # X轴终点
                                    [0, length, 0],  # Y轴终点
                                    [0, 0, length],  # Z轴终点
                                ]).reshape(-1, 3)
    return coordinate_points

def draw_axes(frame, imgpts):
    assert len(imgpts) == 4
    cv2.line(frame, imgpts[0], imgpts[1], (0, 0, 255), 3)
    cv2.line(frame, imgpts[0], imgpts[2], (0, 255, 0), 3)
    cv2.line(frame, imgpts[0], imgpts[3], (255, 0, 0), 3)
    return frame


def drawCube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Add green plane
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # Add box borders
    for i in range(4):
        j = i + 4
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

def draw_o3d_cube(img, pts, color):
    pts = np.int32(pts)

    for i,j in [[0,3],[1,6],[2,5],[7,4],[3,6],[6,4],[4,5],[5,3],[0,2],[2,7],[7,1],[1,0]]:
        img = cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, 3)
    return  img




def draw_predict(region: su.LabeledBoundingBox(), image, color=(0, 255, 0), bbox=True, polyline=False):


    if polyline and region.polygon is not None:

        cv2.polylines(image, [region.polygon.astype('int')], True, color, thickness)
        #cv2.fillPoly(image, [region.polygon.astype('int')], color, cv2.LINE_AA)

    if bbox:
        x1, y1, x2, y2 = region.toint().bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, f"{region.label} {region.score:.2f} ", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, color, thickness, cv2.LINE_AA)

    return image


def generate_distinct_colors(count):
    """生成count个不同的颜色.

    方法来自: http://blog.csdn.net/yhl_leo/article/details/52185581
    """

    assert count < 256, "this method can only generate 255 different colors."

    colors = []
    for i in range(count):
        r, g, b, ii = (0, 0, 0, i)
        for j in range(7):
            str_ii = f"{bin(ii)[2:]:0>8}"[-8:]
            r = r ^ (int(str_ii[-1]) << (7 - j))
            g = g ^ (int(str_ii[-2]) << (7 - j))
            b = b ^ (int(str_ii[-3]) << (7 - j))
            ii = ii >> 3
        colors.append((b, g, r))
    return colors
