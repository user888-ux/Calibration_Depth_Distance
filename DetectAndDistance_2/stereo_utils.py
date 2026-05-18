import cv2
import numpy as np
from collections import deque

class StereoSystem:
    def __init__(self, yaml_file):
        self.load_params(yaml_file)
        self.init_stereo_matcher()

    def load_params(self, yaml_file):
        fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"无法打开YAML文件: {yaml_file}")
        w = int(fs.getNode("image_width").real())
        h = int(fs.getNode("image_height").real())
        self.img_size = (w, h)

        K1 = fs.getNode("camera_matrix_left").mat().astype(np.float64)
        D1 = fs.getNode("dist_coeffs_left").mat().astype(np.float64).reshape(1,-1)
        K2 = fs.getNode("camera_matrix_right").mat().astype(np.float64)
        D2 = fs.getNode("dist_coeffs_right").mat().astype(np.float64).reshape(1,-1)
        R = fs.getNode("rotation_matrix_left_to_right").mat().astype(np.float64)
        T = fs.getNode("translation_vector_left_to_right").mat().astype(np.float64)
        fs.release()

        R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, self.img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        self.validRoi = (roi1, roi2)

        self.map_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, self.img_size, cv2.CV_16SC2)
        self.map_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, self.img_size, cv2.CV_16SC2)

    def init_stereo_matcher(self):
        num_disp = ((self.img_size[0] // 8) + 15) & -16
        self.stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)
        self.stereo.setROI1(self.validRoi[0])
        self.stereo.setROI2(self.validRoi[1])
        self.stereo.setPreFilterCap(31)
        self.stereo.setBlockSize(15)
        self.stereo.setMinDisparity(4)
        self.stereo.setNumDisparities(num_disp)
        self.stereo.setTextureThreshold(50)
        self.stereo.setUniquenessRatio(15)
        self.stereo.setSpeckleWindowSize(100)
        self.stereo.setSpeckleRange(32)
        self.stereo.setDisp12MaxDiff(1)

    def rectify(self, left_img, right_img):
        """输入左右灰度图，返回校正后的图像"""
        left_rect = cv2.remap(left_img, self.map_left[0], self.map_left[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.map_right[0], self.map_right[1], cv2.INTER_LINEAR)
        return left_rect, right_rect

    def compute_depth_map(self, left_rect, right_rect):
        """计算视差图和三维坐标（单位：毫米）"""
        disparity = self.stereo.compute(left_rect, right_rect)
        threeD = cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=True)
        threeD = threeD * 16   # 视差缩放补偿
        return disparity, threeD

    def get_distance_at(self, threeD, x, y):
        """获取指定像素点的距离（米）"""
        if 0 <= x < threeD.shape[1] and 0 <= y < threeD.shape[0]:
            point = threeD[y, x]
            # 如果深度无效的过滤
            if point[2] > 0 and point[2] < 10000:  # 单位：毫米，10米内有效
                dist_mm = np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
                return dist_mm / 1000.0
            else:
                return None
        else:
            return None

    def filter_distance(self,new_dist, buffer, max_rel_error=0.25):
        """
        new_dist: 当前帧计算得到的原始距离（可能为 None 或异常值）
        buffer: 保存有效历史距离的 deque
        max_rel_error: 最大允许相对误差（相对于当前中位数）
        """
        if new_dist is None or new_dist <= 0:
            return None, buffer  # 无效输入，不更新输出

        # 缓冲区还没有足够数据 → 直接接受并加入缓冲区
        if len(buffer) < 3:
            buffer.append(new_dist)
            return new_dist, buffer

        # 计算当前缓冲区的中位数
        sorted_vals = sorted(buffer)
        median = sorted_vals[len(sorted_vals) // 2]

        # 计算相对误差
        rel_error = abs(new_dist - median) / median

        if rel_error <= max_rel_error:
            # 有效：加入缓冲区，并输出平均值（可选）或中位数
            buffer.append(new_dist)
            # 输出缓冲区的中位数（更平滑）或新值
            output = median  # 或者统计平均数
            # 可选：对最新的几个点做平均
            # output = sum(list(buffer)[-3:]) / 3
            return output, buffer
        else:
            # 突变过大，拒绝更新，仍返回上次有效的中位数
            return median, buffer  # 或返回 last_output_distance