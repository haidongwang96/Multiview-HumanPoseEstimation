

## 相机内参标定流程
1. recording.py 分别对已知cam_id进行图像拍摄，保存结果至 data/record/sample_n
2. 在调用 single_camera_calibrate_intrinsic_redo_with_rmse 对进行内参标定
3. 最后保存内参未来使用
4. **Note**: 采用不同分辨率的相机，内参不同 


## Multi Camera Object Tracking

https://github.com/ultralytics/ultralytics/issues/9313

Multi-Camera Live Object Tracking
https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking

Multi Camera People Tracking
https://github.com/hafidh561/multi-camera-people-tracking

CCTVTracker
https://github.com/arvganesh/Multi-Camera-Object-Tracking

Multi-Camera Person Tracking and Re-Identification
https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification

Object detection and tracking with multiple cameras
https://github.com/sourabbsridhar/object-detection-and-tracking-with-multiple-cameras

MTMCT
https://github.com/nolanzzz/mtmct

Multi Camera Object Tracking via Deep Metric Learning
https://github.com/Mhttx2016/Multi-Camera-Object-Tracking-via-Transferring-Representation-to-Top-View

CMU Object Detection & Tracking for Surveillance Video Activity Detection
https://github.com/JunweiLiang/Object_Detection_Tracking