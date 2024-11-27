import cv2
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from collections import defaultdict

import time
import camera
import utility as su
from ultralytics import YOLO


"""
    pose_animate.py 的迭代版
"""
def load_camera_INFO(cam_ids):
    intr = []
    extr = []
    for cam_id in cam_ids:
        intr_i = camera.intr_load(f"data/camera_parameters/camera_{cam_id}_intrinsics.json")
        extr_i = camera.extr_load(f"data/camera_parameters/Extr_C_world_C_{cam_id}.json")
        intr.append(intr_i)
        extr.append(extr_i)
    return intr, extr

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


def track(model, frame, track_history):
    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, half=False, conf=0.5, iou=0.7, classes=[0], verbose=False,
                          tracker="bytetrack.yaml")
    result = results[0] # 因为只输入一张图像
    annotated_frame = result.plot()

    if False and result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()  # boxes.id 可能为None, 由于tracking失效
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    return annotated_frame, result

def process_keypoints(result0, result1, extr=None):
    """
    todo: 多人id
    此函数目前只处理已经匹配了的result0/1
    """
    keypoint0 = result0.keypoints.cpu().numpy()
    keypoint1 = result1.keypoints.cpu().numpy()
    kpts0 = np.squeeze(keypoint0[0].data)
    kpts1 = np.squeeze(keypoint1[0].data)
    duo_pose = su.duo_camera_pose_preprocess(kpts0, kpts1) # 两个视角如果
    if len(duo_pose) >= 5: # 保证有5个keypoint被检测出来
        index = duo_pose[:, 0]
        pts0 = duo_pose[:, 1:3]
        pts1 = duo_pose[:, 3:5]
        point3ds = camera.triangulation(P0, P1, pts0, pts1)
        point3ds = np.multiply(point3ds, 100)
        if extr is not None:
            # p3ds = (extr.R() @ p3ds.T).T + extr.t().T
            point3ds = extr.transform(point3ds)
        point3ds = su.p3d_2_kypt_17format(index, point3ds)
        return point3ds
    else:
        return None

def process_virtual_fence(landmark0, landmark1, extr=None):
    landmark_3d = camera.triangulation(P0, P1, landmark0, landmark1)
    landmark_3d = np.multiply(landmark_3d, 100)
    if extr is not None:
        # p3ds = (extr.R() @ p3ds.T).T + extr.t().T
        landmark_3d = extr.transform(landmark_3d)
    return landmark_3d

def get_kpts3d_center(point3d):
    # p3ds  (17,3)
    point2d = point3d[:, :2]
    x_c, y_c, n = 0, 0, 0
    for p in point2d:
        x,y = p
        if x != 0 and y != 0:
            x_c += x
            y_c += y
            n += 1
    x_c = x_c / n
    y_c = y_c / n
    return [x_c, y_c]

if __name__ == '__main__':


    cam_ids = [2,4]
    intr, extr = load_camera_INFO(cam_ids)

    extr_cam1_to_cam0 = camera.cam1_to_cam2_transformation(extr[0],extr[1])
    extr0_self = camera.get_self_transformation_extrinsic()

    extr_C2_Cw = extr[1].inverse_transformation()

    P0 = camera.ProjectionMatrix(intr[0], extr0_self).get_projection_matrix()
    P1 = camera.ProjectionMatrix(intr[1], extr_cam1_to_cam0).get_projection_matrix()

    model = YOLO("data/weights/yolo11l-pose.pt")

    # cap0 = camera.get_cv2_capture(0)
    # cap1 = camera.get_cv2_capture(1)

    v0 = "data/record_ubuntu/video_0/1732606793310_2.mp4"
    v1 = "data/record_ubuntu/video_0/1732606793310_4.mp4"

    landmark0 = su.read_list_file("data/annotation/mouse_click/landmark_0/1732607373680_2.txt", " ")
    landmark1 = su.read_list_file("data/annotation/mouse_click/landmark_0/1732607373680_4.txt", " ")
    landmark0 = np.array(landmark0, dtype=int)
    landmark1 = np.array(landmark1, dtype=int)

    landmark_3d = process_virtual_fence(landmark0, landmark1, extr_C2_Cw)
    landmark_3d_to_2d = landmark_3d[:,:2] # 2d fence

    cap0 = camera.get_cv2_capture(v0)
    cap1 = camera.get_cv2_capture(v1)

    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open one or both cameras.")
        exit()

    track_history0 = defaultdict(lambda: [])
    track_history1 = defaultdict(lambda: [])

    # 创建绘图窗口和3D坐标轴
    frame_idx = -1
    kypts_3d = []

    out_dir = su.create_asending_folder("data/pose_3d","drawed_frame")
    pose_dir = su.create_asending_folder("data/pose_3d", prefix="pose")
    while True:
        frame_idx = frame_idx + 1
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        # 画出virtual fence
        frame0 = su.draw_fence(frame0, landmark0)
        frame1 = su.draw_fence(frame1, landmark1)

        # Check if frames were captured successfully
        if not ret1 or not ret0:
            print("Error: Could not read frames from one or both cameras.")
            break

        t0 = time.time()
        annotated_frame0, result0 = track(model, frame0, track_history0)
        annotated_frame1, result1 = track(model, frame1, track_history1)

        if result0 is None or result1 is None:
            print(1)
            continue

        p3ds = process_keypoints(result0, result1, extr=extr_C2_Cw)
        if p3ds is None: # 没有有5个keypoint被检测出来
            cv2.imwrite(f"{out_dir}/dframe_{frame_idx}_0.jpg", annotated_frame0)
            cv2.imwrite(f"{out_dir}/dframe_{frame_idx}_1.jpg", annotated_frame1)
            continue
        else:
            kypts_3d.append(p3ds)
            fig, ax = su.pose_3d_plot(p3ds)
            kpt_center = get_kpts3d_center(p3ds)
            isin_zone = is_point_in_quadrilateral(landmark_3d_to_2d, kpt_center)

            if isin_zone:
                vf_color = (1,0,0)
                annotated_frame0 = su.draw_fence(annotated_frame0, landmark0, color=(0,0,255))
                annotated_frame1 = su.draw_fence(annotated_frame1, landmark1, color=(0,0,255))
            else:
                vf_color = (0,1,0)
            #vf_color = (0,1,0)

            su.plot_3d_fence(ax, landmark_3d, vf_color)
            plot_save_name = f"{pose_dir}/pose_3d_{frame_idx}.jpg"
            plt.savefig(plot_save_name)
            plt.close()

            # 合并
            frame = np.hstack((annotated_frame0, annotated_frame1))
            fps = 1 / (time.time() - t0)
            cv2.putText(frame, f"FPS: {fps:.1f} ", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", frame)

            cv2.imwrite(f"{out_dir}/dframe_{frame_idx}_0.jpg", annotated_frame0)
            cv2.imwrite(f"{out_dir}/dframe_{frame_idx}_1.jpg", annotated_frame1)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


    kypts_3d = np.array(kypts_3d)
    print(kypts_3d.shape)
    su.write_pickle_file(kypts_3d,"kpts_3d_video1.pkl")
    #su.write_list_file(kypts_3d,"kpts_3d.txt")


    # pose_dir = su.create_asending_folder("data/pose_3d", prefix="pose")
    # for i, p3ds in enumerate(kypts_3d):
    #
    #     plot_save_name = f"{pose_dir}/pose_3d_{i}.jpg"
    #     print(f"processing {plot_save_name}")
    #     fig = su.pose_3d_plot(p3ds)
    #     plt.savefig(plot_save_name)
    #     plt.close()
