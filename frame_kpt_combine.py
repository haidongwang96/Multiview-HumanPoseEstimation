import os

import cv2
import numpy as np

import camera

i = 1
pose_dir = f"data/pose_3d/pose_{i}"
dframe_dir =f"data/pose_3d/drawed_frame_{i}"
frame_idx = 0

# 创建视频编写器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# size = (width, height)
video = cv2.VideoWriter("out2.mp4", fourcc, 20.0, (1280, 840))

while True:

    dframe0 = f"{dframe_dir}/dframe_{frame_idx}_0.jpg"
    dframe1 = f"{dframe_dir}/dframe_{frame_idx}_1.jpg"

    kpt_frame_path = f"{pose_dir}/pose_3d_{frame_idx}.jpg"
    if not os.path.exists(dframe0) or not os.path.exists(dframe1):
        print("bad",dframe0, dframe1)
        break

    if not os.path.exists(kpt_frame_path):
        kpt_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 255
    else:
        kpt_frame = cv2.imread(kpt_frame_path)

    frame0 = cv2.imread(dframe0)
    frame1 = cv2.imread(dframe1)

    frame0 = cv2.resize(frame0, (640, 360))
    frame1 = cv2.resize(frame1, (640, 360))

    kpt_frame = cv2.resize(kpt_frame, (640, 480))
    white_block =np.zeros((480, 640, 3), dtype=np.uint8) + 255
    up_row = np.hstack((frame0, frame1))
    btm_row = np.hstack((white_block, kpt_frame))
    frame_show = np.vstack((up_row, btm_row))
    cv2.imshow("frame", frame_show)
    video.write(frame_show)
    print(kpt_frame_path, dframe0, dframe1)
    cv2.waitKey(0)
    frame_idx += 1

cv2.destroyAllWindows()
video.release()





