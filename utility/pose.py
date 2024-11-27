import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import utility as su

class Colors:
    """
    Ultralytics color palette https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.

    ## Ultralytics Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "Ultralytics Brand Colors"

        For Ultralytics brand colors see [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand). Please use the official Ultralytics colors for all marketing materials.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


class skeleton_util():
    def __init__(self):
        self.skeleton = [[16, 14],[14, 12],[17, 15],[15, 13],
                         [12, 13],[6, 12],[7, 13],[6, 7],[6, 8],
                         [7, 9],[8, 10],[9, 11],[2, 3],[1, 2],
                             [1, 3],[2, 4],[3, 5],[4, 6],[5, 7]]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

sk_util = skeleton_util()

def duo_camera_pose_preprocess(kpts0, kpts1, conf_thres=0.3):
    # 去去除掉不符合conf threshold的点
    nkpt0, ndim0 = kpts0.shape
    nkpt1, ndim1 = kpts1.shape

    is_pose = nkpt0 ==nkpt1 == 17 and ndim0 == ndim1 ==3
    #assert is_pose
    duo_pose=[]
    for i, (k0, k1) in enumerate(zip(kpts0, kpts1)):
        x0_coord, y0_coord, conf0 = k0[0], k0[1], k0[2]
        x1_coord, y1_coord, conf1 = k1[0], k1[1], k1[2]
        if conf0 < conf_thres or conf1 < conf_thres:
            continue
        duo_pose.append([i, x0_coord, y0_coord, x1_coord, y1_coord])

    return np.array(duo_pose)

def p3d_2_kypt_17format(index, p3ds):
    """
    此函数配合 duo_camera_pose_preprocess生成的index， 将pts转变成17*3的格式
    """
    kpts_3d = []
    for idx, p3d in zip(index, p3ds):
        idx = int(idx)
        while idx > len(kpts_3d):
            kpts_3d.append([0, 0, 0])
        kpts_3d.append(p3d)

    while len(kpts_3d) < 17:
        kpts_3d.append([0, 0, 0])

    return np.array(kpts_3d)


def pose_3d_plot(p3ds):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-600, -100)
    ax.set_ylim(-900, 300)
    ax.set_zlim(-250, 0)

    for i, k in enumerate(p3ds):
        x, y, z = k
        if x == y == z == 0:
            continue
        ax.scatter(x, y, z, c=sk_util.kpt_color[i]/255, marker='o')

    for i, sk in enumerate(sk_util.skeleton):
        pos0 = (p3ds[(sk[0] - 1), 0], p3ds[(sk[0] - 1), 1], p3ds[(sk[0] - 1), 2])
        pos1 = (p3ds[(sk[1] - 1), 0], p3ds[(sk[1] - 1), 1], p3ds[(sk[1] - 1), 2])
        if pos0[0] == pos0[1] == pos0[2] == 0 or pos1[0] == pos1[1] == pos1[2] == 0:
            continue
        ax.plot([pos0[0], pos1[0]], [pos0[1], pos1[1]], [pos0[2], pos1[2]],
                color=sk_util.limb_color[i]/255, linestyle='-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return fig, ax

ldm_line= [[0,1],[1,2],[2,3],[3,0]]
def draw_fence(img, landmarks, color=(0, 255, 0), thickness=2):
    #  链接四个角点的virtual fence
    for landmark in landmarks:
        cv2.circle(img, (landmark[0], landmark[1]), 2, color, thickness)

    for i, line in enumerate(ldm_line):
        pt1 = tuple((landmarks[line[0]][0], landmarks[line[0]][1]))
        pt2 = tuple((landmarks[line[1]][0], landmarks[line[1]][1]))
        cv2.line(img, pt1, pt2, color, thickness)

    return img

def plot_3d_fence(ax, landmarks_3d, color):

    for i, k in enumerate(landmarks_3d):
        x, y, z = k
        if x == y == z == 0:
            continue
        ax.scatter(x, y, z, c=color, marker='o')

    for line in ldm_line:

        pos0 = tuple((landmarks_3d[line[0]][0], landmarks_3d[line[0]][1], landmarks_3d[line[0]][2]))
        pos1 = tuple((landmarks_3d[line[1]][0], landmarks_3d[line[1]][1], landmarks_3d[line[1]][2]))
        ax.plot([pos0[0], pos1[0]], [pos0[1], pos1[1]], [pos0[2], pos1[2]],
                color=color, linestyle='-')

    return ax




