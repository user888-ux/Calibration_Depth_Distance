import cv2
import numpy as np
import time
import math

def load_stereo_params_and_rectify(yaml_file):
    """
    从 YAML 文件加载双目相机标定参数，执行立体校正，返回：
    left_map1, left_map2, right_map1, right_map2, Q, validPixROI1, validPixROI2, image_size
    """
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开 YAML 文件: {yaml_file}")

    # 读取图像尺寸
    image_width = int(fs.getNode("image_width").real())
    image_height = int(fs.getNode("image_height").real())

    # 读取左右相机内参和畸变系数
    left_camera_matrix = fs.getNode("camera_matrix_left").mat()
    left_dist_coeffs = fs.getNode("dist_coeffs_left").mat()
    right_camera_matrix = fs.getNode("camera_matrix_right").mat()
    right_dist_coeffs = fs.getNode("dist_coeffs_right").mat()

    # 读取旋转矩阵和平移向量（左 -> 右）
    R = fs.getNode("rotation_matrix_left_to_right").mat()
    T = fs.getNode("translation_vector_left_to_right").mat()

    fs.release()

    # 转换为 float64 并调整形状
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

    # 计算校正映射表
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_camera_matrix, left_dist_coeffs, R1, P1, size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_camera_matrix, right_dist_coeffs, R2, P2, size, cv2.CV_16SC2
    )

    return (left_map1, left_map2, right_map1, right_map2,
            Q, validPixROI1, validPixROI2, size)

# ========== 主程序 ==========
if __name__ == "__main__":
    # 指定 YAML 文件路径（请根据实际情况修改）
    YAML_FILE = "stereo_calib.yml"

    print("正在加载标定参数并计算校正映射...")
    left_map1, left_map2, right_map1, right_map2, Q, validPixROI1, validPixROI2, img_size = load_stereo_params_and_rectify(YAML_FILE)
    print("加载完成。")
    print(f"图像尺寸（YAML）：{img_size}")
    print(f"有效 ROI 左：{validPixROI1}")
    print(f"有效 ROI 右：{validPixROI2}")

    # 初始化摄像头（假设摄像头输出 1280x480 的拼接图像）
    capture = cv2.VideoCapture(0)
    imageWidth, imageHeight = img_size  # 单个图像的宽度和高度
    print(f"设置摄像头分辨率为：{imageWidth * 2} x {imageHeight}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth * 2)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight)

    actual_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"实际设置的分辨率：{actual_width} x {actual_height}")

    if not capture.isOpened():
        print("错误：无法打开摄像头")
        exit()

    WIN_NAME = 'Deep disp'
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

    # SGBM 立体匹配参数（可根据需要调整）
    blockSize = 8
    img_channels = 3
    stereo = cv2.StereoSGBM_create(
        minDisparity=1,
        numDisparities=64,
        blockSize=blockSize,
        P1=8 * img_channels * blockSize * blockSize,
        P2=32 * img_channels * blockSize * blockSize,
        disp12MaxDiff=-1,
        preFilterCap=140,
        uniquenessRatio=1,
        speckleWindowSize=100,
        speckleRange=100,
        mode=cv2.STEREO_SGBM_MODE_HH
    )

    while True:
        t1 = time.time()
        ret, frame = capture.read()
        if not ret:
            print("未捕获到图像")
            continue

        # 分割左右图像（假定左右拼接，左半部分为左图，右半部分为右图）
        # 注意：原代码中 frame2 切片高度为 640，可能是笔误，这里修正为 imageHeight
        frame1 = frame[0:imageHeight, 0:imageWidth]                     # 左图
        frame2 = frame[0:imageHeight, imageWidth:imageWidth * 2]        # 右图

        imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 立体校正
        img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

        imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
        imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

        # 计算视差图
        disparity = stereo.compute(img1_rectified, img2_rectified)

        # 归一化显示（灰度图）
        disp = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 生成彩色深度图（伪彩色）
        dis_color = cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=255/16), cv2.COLORMAP_JET)

        # 三维重建（注意：disparity 可能为缩放后的值，乘以16以保持与原有逻辑一致）
        threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
        threeD = threeD * 16

        cv2.imshow("depth", dis_color)
        cv2.imshow("left", imageL)
        cv2.imshow("right", imageR)
        cv2.imshow(WIN_NAME, disp)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()