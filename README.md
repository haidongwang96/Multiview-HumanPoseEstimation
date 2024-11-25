# Multiview Human Pose Estimation
## 项目构成
* 相机
  * 单/双录像
  * 单/双拍照
* 校准
  * 棋盘格，aruco
* 坐标变换
  * 内参 
  * 外参
  * 2d 3d之间变换
  * 三角测量
* 模型
  * 检测模型
    * yolov11 (w/o pose)
      * 收集数据
      * 标注数据
      * 划分不同测试集
    * ***TODO*** multi-view re-id 模型 
    * ***TODO*** multi-view tracking
  
* 3D plot
* bash+py args  脚本化全流程
    
## 测试方法
1. dataset cross validation
2. 视频测试
3. 非训练场景泛化测试


## 相机内参标定流程
1. recording.py 分别对已知cam_id进行图像拍摄，保存结果至 data/record/sample_n
2. 在调用 single_camera_calibrate_intrinsic_redo_with_rmse() 对进行内参标定
3. 最后保存内参未来使用
4. **Note**: 采用不同分辨率的相机，内参不同 