

# The COCO keypoints format

1. Nose
2. Left Eye
3. Right Eye
4. Left Ear
5. Right Ear
6. Left Shoulder
7. Right Shoulder
8. Left Elbow
9. Right Elbow
10. Left Wrist
11. Right Wrist
12. Left Hip
13. Right Hip
14. Left Knee
15. Right Knee
16. Left Ankle
17. Right Ankle




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


# cam1_to_cam2_transformation 数学推导

要计算从相机一到相机二的变换矩阵（即求出相机二相对于相机一的旋转矩阵 \( R_c \) 和平移向量 \( t_c \)），可以利用您已知的两个相机各自对同一标记的位姿估计结果 $$ (R_1, t_1) $$ 和 \( (R_2, t_2) \)。下面是详细的数学推导过程：

**1. 定义坐标系与变换关系**

- **标记坐标系（M）**：以标记为参考的坐标系。
- **相机一坐标系（C1）**：相机一的坐标系。
- **相机二坐标系（C2）**：相机二的坐标系。

在OpenCV的`estimatePoseSingleMarkers()`函数中，返回的旋转矩阵 \( R \) 和平移向量 \( t \) 表示从**标记坐标系到相机坐标系**的变换，即：

$$
\mathbf{p}_{C} = R \mathbf{p}_{M} + t
$$

其中，\( \mathbf{p}_{M} \) 是标记坐标系中的点，\( \mathbf{p}_{C} \) 是相应的相机坐标系中的点。

**2. 表达相机与标记的关系**

对于相机一：

$$
\mathbf{p}_{C1} = R_1 \mathbf{p}_{M} + t_1 \quad \text{(1)}
$$

对于相机二：

$$
\mathbf{p}_{C2} = R_2 \mathbf{p}_{M} + t_2 \quad \text{(2)}
$$

**3. 消去标记坐标系中的点**

从方程（1）和（2）中，我们可以解出标记坐标系中的点 \( \mathbf{p}_{M} \)：

从方程（1）：

$$
\mathbf{p}_{M} = R_1^{-1} (\mathbf{p}_{C1} - t_1) \quad \text{(3)}
$$

从方程（2）：

$$
\mathbf{p}_{M} = R_2^{-1} (\mathbf{p}_{C2} - t_2) \quad \text{(4)}
$$

**4. 建立相机坐标系之间的关系**

由于 \( \mathbf{p}_{M} \) 相同，将（3）和（4）等式右边相等：

$$
R_1^{-1} (\mathbf{p}_{C1} - t_1) = R_2^{-1} (\mathbf{p}_{C2} - t_2)
$$

两边同时左乘 \( R_2 \)：

$$
R_2 R_1^{-1} (\mathbf{p}_{C1} - t_1) = \mathbf{p}_{C2} - t_2
$$

因此：

$$
\mathbf{p}_{C2} = R_2 R_1^{-1} (\mathbf{p}_{C1} - t_1) + t_2
$$

**5. 得到相机一到相机二的变换矩阵**

比较上式与相机坐标系之间的标准变换形式 \( \mathbf{p}_{C2} = R_c \mathbf{p}_{C1} + t_c \)，可以得出：

- **旋转矩阵 \( R_c \)**：

$$
R_c = R_2 R_1^{-1}
$$

由于旋转矩阵的逆等于其转置，即 \( R^{-1} = R^\top \)，因此：

$$
R_c = R_2 R_1^\top
$$

- **平移向量 \( t_c \)**：

$$
t_c = -R_c t_1 + t_2
$$

**6. 总结**

通过以上推导，您可以使用以下步骤计算相机一到相机二的变换矩阵：

1. **计算 \( R_c \)**：

   $$
   R_c = R_2 R_1^\top
   $$

2. **计算 \( t_c \)**：

   $$
   t_c = -R_c t_1 + t_2
   $$

**注意事项**：

- 确保旋转矩阵 \( R_1 \) 和 \( R_2 \) 是正交矩阵，可以直接转置求逆。
- 平移向量 \( t_1 \) 和 \( t_2 \) 应该是列向量，维度为 \( 3 \times 1 \)。

通过上述方法，您就可以得到从相机一坐标系到相机二坐标系的变换矩阵 \( (R_c, t_c) \)。

---

**参考公式总结**：

- **相机一到标记的关系**：

  $$
  \mathbf{p}_{C1} = R_1 \mathbf{p}_{M} + t_1
  $$

- **相机二到标记的关系**：

  $$
  \mathbf{p}_{C2} = R_2 \mathbf{p}_{M} + t_2
  $$

- **相机一到相机二的变换**：

  $$
  R_c = R_2 R_1^\top
  $$

  $$
  t_c = -R_c t_1 + t_2
  $$


# project_2d_withdepth_to_3d 数学推导

要将相机一（C1）图像上的某一点（像素坐标）以及已知的深度，转换到相机二（C2）的像素坐标，需要经过以下步骤：

1. **将相机一的像素坐标转换为相机一坐标系下的三维坐标**。

2. **使用相机一到相机二的变换矩阵，将点从相机一坐标系转换到相机二坐标系**。

3. **将相机二坐标系下的三维点投影到相机二的像素坐标系**。

下面详细说明每个步骤的数学过程。

---

### **1. 将相机一的像素坐标转换为相机一坐标系下的三维坐标**

**已知**：

- 相机一的内参矩阵 \( K_1 \)：
  
  $$
  K_1 = \begin{bmatrix}
  f_{x1} & 0 & c_{x1} \\
  0 & f_{y1} & c_{y1} \\
  0 & 0 & 1
  \end{bmatrix}
  $$

- 相机一的像素坐标 \( (u_1, v_1) \)。
- 深度值 \( z_1 \)（相机一坐标系下的深度）。

**步骤**：

1. **计算归一化像平面坐标**：

   $$
   x_{1n} = \frac{u_1 - c_{x1}}{f_{x1}}
   $$
   $$
   y_{1n} = \frac{v_1 - c_{y1}}{f_{y1}}
   $$

2. **得到相机一坐标系下的三维点 \( \mathbf{p}_{C1} \)**：

   $$
   \mathbf{p}_{C1} = \begin{bmatrix}
   x_{1} \\
   y_{1} \\
   z_{1}
   \end{bmatrix} = z_1 \begin{bmatrix}
   x_{1n} \\
   y_{1n} \\
   1
   \end{bmatrix}
   $$



# 四边形内点判断

### 判断点是否在三角形内部

给定三角形的三个顶点A、B、C和待检测的点P，我们可以使用向量叉积的方法判断点P是否在三角形ABC内部。

1. **向量定义：**

   - 向量AB = B - A
   - 向量BC = C - B
   - 向量CA = A - C
   - 向量AP = P - A
   - 向量BP = P - B
   - 向量CP = P - C

2. **计算叉积：**

   - 计算叉积 `Cross1 = AB × AP`
   - 计算叉积 `Cross2 = BC × BP`
   - 计算叉积 `Cross3 = CA × CP`

   在二维空间中，向量叉积的计算公式为：
   $$
   U \times V = u_x \cdot v_y - u_y \cdot v_x
   $$

3. **判断符号：**

   - 如果 `Cross1`、`Cross2`、`Cross3` 的符号相同（都为正或都为负），则点P在三角形内部。
   - 如果任何一个叉积为零，点P在三角形的边上。
   - 如果叉积的符号不一致，点P在三角形外部。

### 应用于四边形

对于四边形ABCD，我们可以将其划分为两个三角形：

- 三角形ABC
- 三角形ACD

然后，分别判断点P是否在其中一个三角形内部。如果点P在任一三角形内部，则点P在四边形内部。







