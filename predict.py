# standard library import
import os
import argparse
# third party import
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import cv2
# local import
from scripts.image_processing import convert_to_array

# argument
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", help="detect mode")
args = vars(ap.parse_args())

capture = cv2.VideoCapture(0)
key = cv2.waitKey(1) & 0xFF
kernel = np.ones((5, 5), np.uint8)
i = 0
center_dots = [] # use to store drawing coordinates
count = None # use to count saved image in save function
label = None # use to draw predicted label in PredictImage class
play_frame = {'Bool': False, 'count': 1} # use to define when to capture drawing
CLEAR_POINTS = {'point1': (0, 0), 'point2': (0, 200), 'point3': (200, 200), 'point4': (200, 0)}
DRAW_POINTS = {'point1': (800, 100), 'point2': (800, 500), 'point3': (1200, 500), 'point4': (1200, 100)}
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
fontScale = 2


def find_blue_color(hsv_frame, original_frame):
    low_red = np.array([112, 100, 20])
    high_red = np.array([129, 255, 255])
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


def capture_drawing():
    global play_frame
    play_frame['count'] = play_frame['count'] + 1
    if play_frame['count'] % 2 == 0:
        play_frame['Bool'] = True
        print("[INFO] resumed capture drawing line .... ")
    elif play_frame['count'] % 2 == 1:
        play_frame['Bool'] = False
        print("[INFO] paused capture drawing line .... ")


def save_image(path_to_count_file, clear_frame, image_to_save, save_path):
    global count
    count_file = open(path_to_count_file, 'r')
    count = count_file.read()
    count_file.close()
    count_file = open(path_to_count_file, 'w')
    count = int(count) + 1
    count_file.write(str(count))
    count_file.close()
    print("[INFO] saving image {} .... ".format(count))
    cv2.imwrite(save_path + "{}.jpg".format(count), image_to_save)
    frame_clone = clear_frame
    center_dots.clear()


class DrawLetter:
    def __init__(self):
        pass

    def draw_with_color(self, frame, frame_clone, white_image):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # find blue color only in draw area
        blue_mask = find_blue_color(hsv_frame[100:500, 800:1200], frame)
        # find contours only in draw area
        (_, contour, _) = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        # Check to see if any contours were found
        if len(contour) > 0:
            contour = sorted(contour, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            # draw circle around contour and draw circle only in draw area
            cv2.circle(frame_clone[100:500, 800:1200], (int(x), int(y)), int(radius), (0, 255, 0), 2)
            # Find center of countour
            moment = cv2.moments(contour)
            # get center dot only in draw area
            center = [(int(moment['m10'] / moment['m00'])) + 800, (int(moment['m01'] / moment['m00'])) + 100]
            cv2.circle(frame_clone, (center[0], center[1]), 12, (255, 255, 255), -1)
            if play_frame['Bool']:
                center_dots.append(center)
        for point in center_dots:
            if (point[0] > 800 and point[0] < 1200) and (point[1] > 100 and point[1] < 500):
                new_x_point = point[0] - 800
                new_y_point = point[1] - 100
                frame_clone = cv2.circle(frame_clone, (point[0] - 5, point[1] - 5), 12, (255, 255, 0), -1)
                white_image = cv2.circle(white_image, (new_x_point - 5, new_y_point - 5), 12, (0, 0, 0), -1)


class PredictImage:
    def __init__(self):
        self.class_labels = []
        self.index = None
        self.predicted_image = None

    def predict_image(self, model_path, class_labels, clear_frame, image_to_save):
        global label
        all_paths = {
            'save_path': "predict_images/input_images/image",
            'image_for_predict_path': 'predict_images/input_images',
            'count_file_path': "count_text/input_predict_count.txt",
            'model_path': model_path,
        }
        # save image to "predict_images/input_images/" directory
        save_image(all_paths['count_file_path'], clear_frame, image_to_save, all_paths['save_path'])

        print("[INFO] processig image .... ")
        # list all image inside given directory and put into list
        image_path = list(paths.list_images(all_paths['image_for_predict_path']))
        # sort all elements inside list
        image_path.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        # get the last image
        image_path = image_path[-1] 
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        # convert to gray scale if draw with hand
        # if mode == "hand": image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype("float") / 255.0
        image = convert_to_array(image)
        image = np.expand_dims(image, axis=0)

        print("[INFO] loading model .... ")
        pre_trained_model = load_model(all_paths['model_path'])

        print("[INFO] predicting .... ")
        self.class_labels = class_labels
        self.predicted_image = pre_trained_model.predict(image)[0]
        percentage = self.predicted_image
        self.index = np.where(self.predicted_image==max(self.predicted_image))
        self.index = self.index[0][0]
        label = self.class_labels[self.index]
        print(f'[INFO] you wrote {label}')
        each_class_percentage = "kor-{:.2f}%, khor-{:.2f}%, kur-{:.2f}%, khur-{:.2f}%".format(percentage.item(0), percentage.item(1), percentage.item(2), percentage.item(3))
        print(each_class_percentage)

    def draw_predict_label(self, frame):
        global label
        if label is not None:   
            cv2.putText(frame, 'You wrote {}'.format(label), (250, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame


def create_white_image(size=[300, 300]):
    image = np.zeros(size, dtype=np.uint8)
    image.fill(255)
    return image


def main():
    print("[INFO] accessing camera .... ")
    print("[INFO] opening camera .... ")
    global i, count, key, play_frame
    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        clear_frame_clone = frame.copy()
        # clear_frame_clone = draw_clear_area(clear_frame_clone)
        clear_frame_clone = draw_area(clear_frame_clone)
        clear_frame_clone = draw_save_count(clear_frame_clone)
        clear_frame_clone = PredictImage().draw_predict_label(clear_frame_clone)
        frame_clone = clear_frame_clone.copy()
        white_image = create_white_image(size=[400, 400])

        # drawing on air
        if args["mode"] == "color":
            DrawLetter().draw_with_color(frame, frame_clone, white_image)
        # elif args["mode"] == "hand":
        #     DrawLetter().draw_with_hand(frame, frame_clone, "model/hand_neural_net.hdf5")

        cv2.imshow("original", frame_clone)
        cv2.imshow("white", white_image)

        key = cv2.waitKey(1) & 0xFF

        # keys events
        if key == ord('s'):
            save_images_path = "../khmer_letter_dataset/khmer_letters_air/original/5_ngur_original/ngur"
            save_image("count_text/count.txt", clear_frame_clone, white_image, save_images_path)
        elif key == ord('p'):
            model_path = "model/k_neural_net__on_air.hdf5"
            class_labels = ['kor', 'khor', 'kur', 'khur']
            predict = PredictImage()
            predict.predict_image(model_path, class_labels, clear_frame_clone, white_image)
        elif key == ord('a'):
            capture_drawing()
        elif key == ord('c'):
            for point in center_dots:
                clear_frame_clone = draw_clear_area(clear_frame_clone, point[0], point[1])
                frame_clone = clear_frame_clone
                center_dots.clear()
                print("Image's cleared")
                break
        elif key == ord('q'):
            print("[INFO] closing camera .... ")
            print("Closed")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("[INFO] Runing predict module .... ")
    main()