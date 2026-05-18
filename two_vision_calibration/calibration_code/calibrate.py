import numpy as np
import cv2 as cv
import glob
import os

# ================= 代码配置区（请根据你的实际情况修改！） =================
CHECKERBOARD = (7,7)          # 内角点尺寸，根据你的棋盘格修改
SQUARE_SIZE = 23              # 方格边长 (mm)
CALIBRATION_LEFT_IMGS_PATH = "../my_left/left*.jpg"
CALIBRATION_RIGHT_IMGS_PATH = "../my_right/right*.jpg"
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
# ====================================================================

# --- 1. 准备3D世界坐标点 ---
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

objpoints = []
imgpoints_left = []
imgpoints_right = []

images_left = sorted(glob.glob(CALIBRATION_LEFT_IMGS_PATH))
images_right = sorted(glob.glob(CALIBRATION_RIGHT_IMGS_PATH))

if not images_left or not images_right:
    print(f"错误：在指定路径下没有找到图片。请检查路径和文件名中的 'left' 和 'right' 前缀。")
    exit()

if len(images_left) != len(images_right):
    print(f"警告：左图数量({len(images_left)})和右图数量({len(images_right)})不匹配，请检查！")
    exit()

print(f"成功找到 {len(images_left)} 组图片。开始检测角点...")

# --- 2. 遍历所有图片对，检测棋盘格角点 ---
for img_left_file, img_right_file in zip(images_left, images_right):
    img_left = cv.imread(img_left_file)
    img_right = cv.imread(img_right_file)
    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    ret_left, corners_left = cv.findChessboardCornersSB(gray_left, CHECKERBOARD, None)
    ret_right, corners_right = cv.findChessboardCornersSB(gray_right, CHECKERBOARD, None)

    if ret_left and ret_right:
        objpoints.append(objp)
        corners_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

        cv.drawChessboardCorners(img_left, CHECKERBOARD, corners_left, ret_left)
        cv.drawChessboardCorners(img_right, CHECKERBOARD, corners_right, ret_right)
        cv.imshow('Left Image', img_left)
        cv.imshow('Right Image', img_right)
        cv.waitKey(1000)
    else:
        print(f"警告：在图片对 {os.path.basename(img_left_file)} 和 {os.path.basename(img_right_file)} 中未能同时检测到角点，已跳过。")

cv.destroyAllWindows()
print(f"成功找到并使用了 {len(objpoints)} 组有效的图片对。")

# --- 3. 执行双目标定 ---
if len(objpoints) < 10:
    print("错误：有效的图片对数量少于10组，标定结果可能不准确，建议至少使用10-20组图片。")
    exit()

img_shape = gray_left.shape[::-1]   # (width, height)

print("正在执行单目标定...")
ret_left, mtx_left, dist_left, _, _ = cv.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
ret_right, mtx_right, dist_right, _, _ = cv.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)
print("单目标定完成。")

flags = cv.CALIB_USE_INTRINSIC_GUESS
print("正在执行双目标定，这可能需要一点时间...")
ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left, mtx_right, dist_right,
    img_shape, criteria=criteria, flags=flags
)
print("双目标定完成。")

# --- 4. 立体校正 ---
print("正在计算立体校正参数...")
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right, img_shape, R, T, flags=0, alpha=0
)

map_left_x, map_left_y = cv.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, img_shape, cv.CV_32FC1)
map_right_x, map_right_y = cv.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, img_shape, cv.CV_32FC1)

# ================= 保存标定结果为 .npz =================
np.savez(
    'stereo_calibration_result.npz',
    mtx_left=mtx_left, dist_left=dist_left,
    mtx_right=mtx_right, dist_right=dist_right,
    R=R, T=T, E=E, F=F,
    R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
    map_left_x=map_left_x, map_left_y=map_left_y,
    map_right_x=map_right_x, map_right_y=map_right_y
)

# ================= 新增：保存标定结果为 .yml（符合 OpenCV 格式） =================
print("正在保存标定结果至 'stereo_calibration_result.yml'...")

# 使用 OpenCV 的 FileStorage 写入 YAML 文件
fs = cv.FileStorage('stereo_calibration_result.yml', cv.FILE_STORAGE_WRITE)

# 写入图像尺寸
fs.write('image_width', img_shape[0])
fs.write('image_height', img_shape[1])

# 写入左相机内参矩阵 (3x3)
fs.write('camera_matrix_left', mtx_left)
# 写入左相机畸变系数 (1x5)
fs.write('dist_coeffs_left', dist_left.reshape(1, -1))

# 写入右相机内参矩阵
fs.write('camera_matrix_right', mtx_right)
# 写入右相机畸变系数
fs.write('dist_coeffs_right', dist_right.reshape(1, -1))

# 写入旋转矩阵 (左->右)
fs.write('rotation_matrix_left_to_right', R)
# 写入平移向量 (左->右)，作为 3x1 矩阵
fs.write('translation_vector_left_to_right', T.reshape(3, 1))

# 写入本质矩阵 E 和基础矩阵 F
fs.write('essential_matrix', E)
fs.write('fundamental_matrix', F)

# 可选：也可以保存重投影矩阵 Q，方便后续使用
fs.write('Q_matrix', Q)

# 关闭文件
fs.release()

# ================= 打印标定结果摘要 =================
print("\n================ 双目相机标定结果 ================")
print(f"平均重投影误差 (RMS): {ret} 像素，越小越好，通常 < 0.5")
print("\n左相机内参矩阵 (Intrinsic Matrix):\n", mtx_left)
print("\n右相机内参矩阵 (Intrinsic Matrix):\n", mtx_right)
print("\n右相机相对于左相机的旋转矩阵 (Rotation matrix R):\n", R)
print("\n右相机相对于左相机的平移向量 (Translation vector T):\n", T.ravel())
print("\n重投影矩阵 Q (用于2D到3D转换):\n", Q)
print("\n所有详细参数已保存至 'stereo_calibration_result.npz' 和 'stereo_calibration_result.yml' 文件。")