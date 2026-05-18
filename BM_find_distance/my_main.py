# -*- coding: utf-8 -*-

import numpy as np
import cv2
import random
import math

def load_stereo_params_and_rectify(yaml_file):
    """从YAML文件加载立体相机参数，执行立体校正，返回映射表和Q矩阵等"""
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开YAML文件: {yaml_file}")

    # 读取图像尺寸
    image_width = int(fs.getNode("image_width").real())
    image_height = int(fs.getNode("image_height").real())

    # 读取左右相机内参和畸变系数
    left_camera_matrix = fs.getNode("camera_matrix_left").mat()
    left_dist_coeffs = fs.getNode("dist_coeffs_left").mat()
    right_camera_matrix = fs.getNode("camera_matrix_right").mat()
    right_dist_coeffs = fs.getNode("dist_coeffs_right").mat()

    # 读取旋转矩阵和平移向量（左->右）
    R = fs.getNode("rotation_matrix_left_to_right").mat()
    T = fs.getNode("translation_vector_left_to_right").mat()

    fs.release()

    # 确保数据类型为float64
    left_camera_matrix = left_camera_matrix.astype(np.float64)
    left_dist_coeffs = left_dist_coeffs.astype(np.float64).reshape(1, -1)
    right_camera_matrix = right_camera_matrix.astype(np.float64)
    right_dist_coeffs = right_dist_coeffs.astype(np.float64).reshape(1, -1)
    R = R.astype(np.float64)
    T = T.astype(np.float64)

    size = (image_width, image_height)

    # 立体校正
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        left_camera_matrix, left_dist_coeffs,
        right_camera_matrix, right_dist_coeffs,
        size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # 计算左右映射表（用于remap）
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_camera_matrix, left_dist_coeffs, R1, P1, size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_camera_matrix, right_dist_coeffs, R2, P2, size, cv2.CV_16SC2
    )

    return (left_map1, left_map2, right_map1, right_map2,
            Q, validPixROI1, validPixROI2, size)

# ==================== 主程序 ====================
# 指定YAML文件路径（根据实际情况修改）
YAML_FILE = "stereo_calibration_result.yml"

# 加载并校正
print("正在加载标定参数并计算校正映射...")
(left_map1, left_map2, right_map1, right_map2,
 Q, validPixROI1, validPixROI2, img_size) = load_stereo_params_and_rectify(YAML_FILE)
print("加载完成。")

# 打开摄像头（根据实际设备调整索引，此处使用1）
cap = cv2.VideoCapture(0)
# 设置分辨率，注意双目相机通常左右拼接为1280x480
cap.set(3, 1280)
cap.set(4, 480)

# 鼠标回调函数（用于显示距离）
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        # 获取该点的三维坐标（单位：毫米？实际取决于Q矩阵和视差缩放）
        # 注意：这里threeD[y][x][2]是深度值（毫米），需要转换为米
        distance = math.sqrt(threeD[y][x][0]**2 + threeD[y][x][1]**2 + threeD[y][x][2]**2)
        distance = distance / 1000.0  # 毫米 -> 米
        print(f"距离：{distance:.3f} m")

WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

# 立体匹配参数（可自行调整）
numberOfDisparities = ((640 // 8) + 15) & -16   # 保证是16的倍数

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)
stereo.setROI1(validPixROI1)
stereo.setROI2(validPixROI2)
stereo.setPreFilterCap(31)
stereo.setBlockSize(15)
stereo.setMinDisparity(4)
stereo.setNumDisparities(numberOfDisparities)
stereo.setTextureThreshold(50)
stereo.setUniquenessRatio(15)
stereo.setSpeckleWindowSize(100)
stereo.setSpeckleRange(32)
stereo.setDisp12MaxDiff(1)

print("开始实时测距，按 'q' 退出...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取图像帧")
        break

    # 分割左右图像（假设图像为左右拼接，左半部分640x480，右半部分640x480）
    frame1 = frame[0:480, 0:640]      # 左图
    frame2 = frame[0:480, 640:1280]   # 右图

    # 转为灰度图
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 立体校正（重映射）
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 计算视差图
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # 归一化显示（仅用于展示）
    disp_display = cv2.normalize(disparity, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 三维重建（注意：disparity是16倍缩放后的值，reprojectImageTo3D内部会除以16？）
    # OpenCV的reprojectImageTo3D默认期望视差是原始视差（浮点），若disparity是整数缩放，需先转换
    # 此处保持与原代码一致：直接传入disparity，然后结果乘以16
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    threeD = threeD * 16   # 原代码中的缩放，保持距离计算逻辑

    # 设置鼠标回调
    cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, threeD)

    # 显示图像
    cv2.imshow("left", frame1)
    cv2.imshow("right", frame2)
    cv2.imshow(WIN_NAME, disp_display)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()