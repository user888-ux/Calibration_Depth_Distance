import time

import cv2
from stereo_utils import StereoSystem
from color_detector import detect_color_objects
from collections import deque

YAML_FILE = "../Depth/stereo_calib.yml"

data_counts=0

print("初始化双目系统...")
stereo_sys = StereoSystem(YAML_FILE)

cap = cv2.VideoCapture(0) # 不行就改成0
cap.set(3, 1280)
cap.set(4, 480)

# 初始化滤波队列
distance_buffer = deque(maxlen=10)   # 保留最近10个有效距离
last_output_distance = None           # 上一次最终输出的距离

# 打开结果文件（追加模式，若每次运行需要新文件可改为 'w'）
result_file = open("result.txt", "w", encoding="utf-8")

print("开始实时检测，按 'q' 退出...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 分割左右图
        left_img = frame[:, :640]
        right_img = frame[:, 640:]

        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # 校正
        left_rect, right_rect = stereo_sys.rectify(gray_left, gray_right)

        # 计算深度图
        disparity, threeD = stereo_sys.compute_depth_map(left_rect, right_rect)

        # 颜色检测（在左原图上）
        left_disp = left_img.copy()
        centers = detect_color_objects(left_disp, min_area=500, color='green')  # 这里面积过滤阈值、识别颜色

        # 输出距离
        for (cx, cy) in centers:
            dist = stereo_sys.get_distance_at(threeD, cx, cy)

            filtered_dist, distance_buffer = stereo_sys.filter_distance(dist, distance_buffer)
            if filtered_dist is not None:
                msg = f"{dist:.3f}m -> {filtered_dist:.3f}m"
                print(msg)
                result_file.write(msg + "\n")
                result_file.flush()  # 实时写入磁盘
                # 提示数据收集了几组
                data_counts+=1
                if(data_counts % 500 == 0):
                    print("已经收集%d组数据" % data_counts)
                    print("按 'q' 退出 或者 等待5秒自动继续收集")
                    time.sleep(5)
                cv2.putText(left_disp, f"{filtered_dist:.2f}m", (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                # 显示上次有效距离
                if last_output_distance:
                    msg = f"坐标({cx},{cy}) 距离: {dist:.3f} m"
                    print(msg)
                    result_file.write(msg + "\n")
                    result_file.flush()
                    cv2.putText(left_disp, f"{dist:.2f}m", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 显示
        disp_show = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("Detection", left_disp)
        cv2.imshow("Disparity", disp_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    result_file.close()
    cap.release()
    cv2.destroyAllWindows()