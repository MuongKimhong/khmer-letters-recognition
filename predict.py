import cv2
import numpy as np

capture = cv2.VideoCapture(0)


def find_red_color(hsv_frame, original_frame):
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    red_spot = {'red': red, 'red_mask': red_mask}
    return red_spot


while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = find_red_color(hsv_frame, frame)
    coordinate = cv2.findNonZero(red['red_mask'])
    if coordinate is not None:
        print("Coordinate (x:{}, y:{})".format(coordinate[0][0][0], coordinate[0][0][1]))
        frame = cv2.circle(frame, (coordinate[0][0][0], coordinate[0][0][1]), radius=10, color=(0, 0, 255), thickness=-1)

    cv2.imshow('red', red['red'])
    cv2.imshow('original', frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()