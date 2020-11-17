import cv2
import numpy as np

print("[INFO] accessing camera .... ")
print("[INFO] opening camera .... ")
capture = cv2.VideoCapture(0)
kernel = np.ones((5, 5), np.uint8)
center_dots = []
count = None
i = 1
CLEAR_POINTS = {'point1': (0, 0), 'point2': (0, 200), 'point3': (200, 200), 'point4': (200, 0)}
DRAW_POINTS = {'point1': (800, 100), 'point2': (800, 500), 'point3': (1200, 500), 'point4': (1200, 100)}
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
fontScale = 2


def find_blue_color(hsv_frame, original_frame):
    low_red = np.array([110, 100, 20])
    high_red = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_red, high_red)
    blue_mask = cv2.erode(blue_mask, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)
    return blue_mask


def draw_clear_area(frame, point0=None, point1=None):
    if (point0 is not None) and (point1 is not None):
        if point0 <= 200 and point1 <= 200: 
            cv2.rectangle(frame, CLEAR_POINTS['point1'], CLEAR_POINTS['point3'], (221, 255, 161), -1)
            cv2.putText(frame, 'Clear', (20, 120), font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, CLEAR_POINTS['point1'], CLEAR_POINTS['point3'], (0, 255, 0), 5)
        cv2.putText(frame, 'Clear', (20, 120), font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
    return frame


def draw_area(frame):
    cv2.line(frame, DRAW_POINTS['point1'], DRAW_POINTS['point2'], (0, 255, 0), 6)
    cv2.line(frame, DRAW_POINTS['point2'], DRAW_POINTS['point3'], (0, 255, 0), 6)
    cv2.line(frame, DRAW_POINTS['point3'], DRAW_POINTS['point4'], (0, 255, 0), 6)
    cv2.line(frame, DRAW_POINTS['point4'], DRAW_POINTS['point1'], (0, 255, 0), 6)
    return frame


def draw_save_count(frame):
    global count
    if count is not None:
        cv2.putText(frame, 'image {} saved'.format(count), (250, 120), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def save_image(path_to_count_file, clear_frame, image_to_save):
    count_file = open(path_to_count_file, 'r')
    count = count_file.read()
    count_file.close()
    count_file = open(path_to_count_file, 'w')
    count = int(count) + 1
    count_file.write(str(count))
    count_file.close()
    print("[INFO] saving image {} .... ".format(count))
    cv2.imwrite("../saved_images/image{}.jpg".format(count), image_to_save)
    frame_clone = clear_frame
    center_dots.clear()


def create_white_image(size=[300, 300]):
    image = np.zeros(size, dtype=np.uint8)
    image.fill(255)
    return image


def main():
    global i, count
    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        clear_frame_clone = frame.copy()
        clear_frame_clone = draw_clear_area(clear_frame_clone)
        clear_frame_clone = draw_area(clear_frame_clone)
        clear_frame_clone = draw_save_count(clear_frame_clone)
        frame_clone = clear_frame_clone.copy()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = find_blue_color(hsv_frame, frame)

        white_image = create_white_image(size=[400, 400])
        # Find contours
        (_, contour, _) = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        # Check to see if any contours were found
        if len(contour) > 0:
            contour = sorted(contour, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            # draw circle around contour
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            # Find center of countour
            moment = cv2.moments(contour)
            center = [int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00'])]
            center_dots.append(center)

        for point in center_dots:
            if (point[0] > 800 and point[0] < 1200) and (point[1] > 100 and point[1] < 500):
                new_x_point = point[0] - 800
                new_y_point = point[1] - 100
                cv2.circle(frame_clone, (point[0] - 30, point[1] - 30), 12, (255, 255, 0), -1)
                cv2.circle(white_image, (new_x_point - 30, new_y_point - 30), 12, (0, 0, 0), -1)
                # cv2.line(frame_clone, (point[0] - 30, point[1] - 30), (point[0], point[1]), (255, 255, 0), 18)
                # cv2.line(white_image, (new_x_point - 7, new_y_point - 7), (new_x_point, new_y_point), (0, 0, 0), 18)

            if point[0] <= 200 and point[1] <= 200:
                clear_frame_clone = draw_clear_area(clear_frame_clone, point[0], point[1])
                frame_clone = clear_frame_clone
                center_dots.clear()
                print("Cleared")
                continue

        cv2.imshow("original", frame_clone)
        cv2.imshow("white", white_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            save_image("count.txt", clear_frame_clone, white_image)
        elif key == ord('q'):
            print("break")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()