import cv2
import os
print(-1)
cap = cv2.VideoCapture(0)
print(0)
# set the video frame width and height
cap.set(3,1280)
print(1)
cap.set(4,480)
print(2)
if not os.path.exists("left"):
    os.makedirs("left")

if not os.path.exists("right"):
    os.makedirs("right")

i = 1
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        # split the frame into left and right
        left_frame = frame[:, :640, :]
        right_frame = frame[:, 640:, :]
        print(3)
        cv2.imshow("Left Camera", left_frame)
        cv2.imshow("Right Camera", right_frame)
        print(4)
        key = cv2.waitKey(1) & 0xFF
        print(5)
        if key == ord('q'):
            break
        elif key == ord('s'):
            left_filename = "left" + str(i) + ".jpg"
            right_filename = "right" + str(i) + ".jpg"
            cv2.imwrite("../my_left/" + left_filename, left_frame)
            cv2.imwrite("../my_right/" + right_filename, right_frame)
            print("Image saved! left image:", left_filename)
            print("Image saved! right image:", right_filename)
            i += 1

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    cap.release()
