import cv2
import os
import numpy as np

from pose_triangulation import keypoint0

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from collections import defaultdict

import time
import camera
import utility as su
from ultralytics import YOLO


"""
todo:   1. 3D PLOT multi-color (DONE)
        2. save images with plots 
"""


pack =su.read_json_file("pack.json")
extr0_self = camera.Extrinsic(np.array(pack["extr0_self_r"]), np.array(pack["extr0_self_t"]))
intr0 = camera.Intrinsics(np.array(pack["intr0"]))
extr_cam1_to_cam0 = camera.Extrinsic(np.array(pack["extr_cam1_to_cam0_r"]), np.array(pack["extr_cam1_to_cam0_t"]))
intr1 = camera.Intrinsics(np.array(pack["intr1"]))

P0 = camera.ProjectionMatrix(intr0, extr0_self).get_projection_matrix()
P1 = camera.ProjectionMatrix(intr1, extr_cam1_to_cam0).get_projection_matrix()


def track(model, frame, track_history):
    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, half=True, conf=0.5, iou=0.7, classes=[0], verbose=True,
                          tracker="bytetrack.yaml")

    if results is not None:
        annotated_frame = results[0].plot()
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
    else:
        annotated_frame = frame

    return annotated_frame, results

def process_keypoints(result0,result1):
    """
    todo: 多人id
    """
    keypoint0 = result0[0].keypoints.cpu().numpy()
    keypoint1 = result1[0].keypoints.cpu().numpy()
    kpts0 = np.squeeze(keypoint0[0].data)
    kpts1 = np.squeeze(keypoint1[0].data)
    duo_pose = su.duo_camera_pose_preprocess(kpts0, kpts1)
    index = duo_pose[:, 0]
    pts0 = duo_pose[:, 1:3]
    pts1 = duo_pose[:, 3:5]
    p3ds = camera.triangulation(P0, P1, pts0, pts1)
    p3ds = np.multiply(p3ds, 100)
    p3ds = su.p3d_2_kypt_17format(index, p3ds)
    return p3ds


model = YOLO("data/weights/yolo11n-pose.pt")

# cap0 = camera.get_cv2_capture(0)
# cap1 = camera.get_cv2_capture(1)

v0 = "data/record/video_0/1732262179527_0.mp4"
v1 = "data/record/video_0/1732262179527_1.mp4"

cap0 = camera.get_cv2_capture(v0)
cap1 = camera.get_cv2_capture(v1)

if not cap0.isOpened() or not cap1.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

track_history0 = defaultdict(lambda: [])
track_history1 = defaultdict(lambda: [])


# 创建绘图窗口和3D坐标轴
frame_idx = 0
os.makedirs("pose", exist_ok=True)

kypts_3d = []

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    # Check if frames were captured successfully
    if not ret1 or not ret0:
        print("Error: Could not read frames from one or both cameras.")
        break

    t0 = time.time()

    annotated_frame0, result0 = track(model, frame0, track_history0)
    annotated_frame1, result1 = track(model, frame1, track_history1)
    # if result0 is None or result1 is None:
    #     print(1)
    #     continue

    p3ds = process_keypoints(result0, result1)
    kypts_3d.append(p3ds)
    # plot_save_name = f"pose/pose_3d_{frame_idx}.png"
    # su.pose_3d_plot(p3ds, frame_idx)
    #
    # plt.savefig(plot_save_name)
    #plt.show()
    frame_idx += 1

    # 合并
    frame = np.hstack((annotated_frame0, annotated_frame1))
    fps = 1 / (time.time() - t0)
    cv2.putText(frame, f"FPS: {fps:.1f} ", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Display the annotated frame
    cv2.imshow("YOLO11 Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# 释放VideoCapture对象
cap0.release()
cap1.release()
cv2.destroyAllWindows()

kypts_3d = np.array(kypts_3d)
su.write_pickle_file(kypts_3d,"kpts_3d.pkl")