import cv2
import numpy as np

def detect_color_objects(frame, min_area, color='red'):
    """
    检测指定颜色的物体中心，返回坐标列表。
    color: 'red', 'blue', 'green'
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if color == 'red':
        #                   如果灵敏度不够就将 100 调小
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([179, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                              cv2.inRange(hsv, lower2, upper2))
    elif color == 'blue':
        lower = np.array([100, 100, 100])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color == 'green':
        lower = np.array([40, 50, 50])
        upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        raise ValueError("Unsupported color")

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
            # 在图像上绘制（可选）
            cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
    return centers