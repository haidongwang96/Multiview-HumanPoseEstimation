import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Demos.FileSecurityTest import permissions

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import camera
import utility as su
from ultralytics import YOLO



pack =su.read_json_file("pack.json")
extr0_self = camera.Extrinsic(np.array(pack["extr0_self_r"]), np.array(pack["extr0_self_t"]))
intr0 = camera.Intrinsics(np.array(pack["intr0"]))
extr_cam1_to_cam0 = camera.Extrinsic(np.array(pack["extr_cam1_to_cam0_r"]), np.array(pack["extr_cam1_to_cam0_t"]))
intr1 = camera.Intrinsics(np.array(pack["intr1"]))

P0 = camera.ProjectionMatrix(intr0, extr0_self).get_projection_matrix()
P1 = camera.ProjectionMatrix(intr1, extr_cam1_to_cam0).get_projection_matrix()



model = YOLO("data/weights/yolo11n-pose.pt")

image_folder_path ="data/record/pose_0"
frame0 = cv2.imread(f"{image_folder_path}/1732004879968_0.jpg")
frame1 = cv2.imread(f"{image_folder_path}/1732004879968_1.jpg")

results = model([frame0, frame1], half=True, conf=0.5, iou=0.7, classes=[0], verbose=False,)
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     keypoints = result.keypoints.cpu().numpy()  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.save()
    #result.show()  # display to screen

keypoint0 = results[0].keypoints.cpu().numpy()
keypoint1 = results[1].keypoints.cpu().numpy()
kpts0 = np.squeeze(keypoint0.data)
kpts1 = np.squeeze(keypoint1.data)


duo_pose = su.duo_camera_pose_preprocess(kpts0, kpts1)
index = duo_pose[:,0]
pts0 = duo_pose[:,1:3]
pts1 = duo_pose[:,3:5]

su.print_block()
p3ds = camera.triangulation(P0, P1, pts0, pts1)
p3ds = su.p3d_2_kypt_17format(index, p3ds)
fig = plt.figure()
su.pose_3d_plot(p3ds, fig)
plt.show()







