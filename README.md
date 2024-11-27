# Multiview Human Pose Estimation


## 任务模块
1. 2D Object detection 
2. Cross frame Re-identification (Tracking)
3. 3D vision
Camera calibration
Triangulation


https://www.bilibili.com/video/BV1mT4y1o7Q2?spm_id_from=333.788.videopod.sections&vd_source=a45e2faca7b4f53ef2dd177795c18f34


## todo:
* [pose_animate.py](pose_animate.py) process_keypoints 多人id的处理，multiframe id的reid 之后进行3d triangulation
* keypoint 滤波

## task:
1. 收集数据，标注数据
   1. 录制数据时，拍摄一组 chessboard 图像（n)
   2. 拍摄一组aruco 图像（1）
   3. 记录图像保存位置（路径名）
   4. 放置四个角点（aruco) 拍摄landmark

## 项目构成
* 相机
  * 单/双录像 recording.py
  * 单/双拍照 recording.py
* 校准
  * 棋盘格，aruco
* 坐标变换
  * 内参 
  * 外参
  * 2d 3d之间变换
  * 三角测量
  * ***(TODO)*** 利用aruco确立世界坐标系
* 模型
  * 检测模型
    * yolov11 (w/o pose)
    * ***(TODO)*** multi-view re-id 模型 
    * ***(TODO)*** multi-view tracking
* 可视化
  *  3D plot
  *  virtual fence 绘制
  * ***(TODO)*** 整合场景 plot
* 系统化
  * ***(TODO)*** bash+py args  脚本化全流程
    
## 测试方法
1. dataset cross validation
   * 收集数据
   * 标注数据
   * 划分不同测试集
2. 视频测试
3. 非训练场景泛化测试


## 相机内参标定流程
1. [recording.py](recording.py) 分别对已知cam_id进行图像拍摄，保存结果至 data/record/chessboard_n
2. [calibration.py](calibration.py) 在调用 calibrate_both_intrinsics(sample_folder) 对进行内参标定
4. **Note**: 采用不同分辨率的相机，内参不同 

## 相机外参标定流程
1. [aruco_test_stream_duo.py](aruco_test_stream_duo.py)  可以对双相机进行aruco进行展示，确保相机都可以对aruco码进行检测
2. 之后调用recording，拍摄双相机aruco图片
3. 使用[check_duo_calibration.py](check_duo_calibration.py) 对拍摄好的aruco进行分析，使用卷尺查看是否正确，并保存双相机之间的外参 
4. **Note**: 注意两个相机的顺序

## 划定 virtual fence边界
* 计算virtual fence边界


## 数据收集
1. 架设双相机，使用[recording.py](recording.py)进行拍摄，保存不同照片
2. 进行yolo格式的数据标注