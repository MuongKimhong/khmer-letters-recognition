# standard library import
import os

# third party import
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import cv2


class ImageProcessing:
    def __init__(self, input_img_path, output_img_path):
        self.input_img_path  = input_img_path
        self.output_img_path = output_img_path
        # turn all image paths to list
        self.image_paths     = list(paths.list_images(self.input_img_path))

    def rename_image(self, new_name, file_type="jpg"):
        if not self.image_paths:
            raise Exception(f"{self.input_img_path} is an empty directory")
        else:
            for (i, image_path) in enumerate(self.image_paths):
                if len(self.image_paths) == 1:
                    output_path = self.output_img_path + "/" + f'{new_name}' + f'.{file_type}'
                elif len(self.image_paths) > 1:
                    output_path = self.output_img_path + "/" f'{new_name}' + f'{i+1}' + f'.{file_type}'
                os.rename(image_path, output_path)
            print("Done.")

    def crop(self, image, top, bottom, left, right):
        image_height, image_width = image.shape[:2]
        image_layout  = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
        cropped_image = image[image_layout['top']:(image_height - image_layout['bottom']), image_layout['left']:(image_width - image_layout['right'])]
        return cropped_image

    def resize_image(self, size, channels, rename=False, new_name=None, file_type="jpg", crop_image=False, top=None, bottom=None, left=None, right=None, padding_color=0):
        for (index, image_path) in enumerate(self.image_paths):
            padding = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
            if channels != 3 and channels != 1:
                raise Exception("valid values for channels is 1 and 3 but {} is provided".format(channels))
            elif channels == 1:
                image = cv2.imread(image_path, 0)
            elif channels == 3:
                image = cv2.imread(image_path)
            height, width = image.shape[:2]
            new_height, new_width = size

            # check if user want to rename image during resizing
            if rename and (new_name is None):
                raise Exception("rename is set to True but no new name provided")
            elif rename and (new_name is not None):
                self.rename_image(new_name, file_type)

            # check if user want to crop image during resizing
            if crop_image and ((top is None) or (bottom is None) or (left is None) or (right is None)):
                for (none_position_key, none_position_val) in {'top': top, 'bottom': bottom, 'left': left, 'right': right}.items():
                    if none_position_val is None:
                        raise Exception("crop_image is set to True but {} is None".format(none_position_key))
            elif crop_image and top and bottom and left and right:
                cropped_image = self.crop(image, top, bottom, left, right)
                cropped_image_height, cropped_image_width = cropped_image.shape[:2]

                if (cropped_image_height > new_height) or (cropped_image_width > new_width):
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC

                print("[INFO] resizing images {}/{}".format(index+1, len(self.image_paths)))
                resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=interpolation)
                resized_image = cv2.copyMakeBorder(resized_image, padding['top'], padding['bottom'], padding['left'], padding['right'], borderType=cv2.BORDER_CONSTANT)
            elif not crop_image:
                if (height > new_height) or (width > new_width):
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC

                print("[INFO] resizing images {}/{}".format(index+1, len(self.image_paths)))
                
                if len(image.shape) == 3 and not isinstance(padding_color, (list, tuple, np.ndarray)):
                    padding_color = [padding_color] * 3
                
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
                resized_image = cv2.copyMakeBorder(resized_image, padding['top'], padding['bottom'], padding['left'], padding['right'], borderType=cv2.BORDER_CONSTANT, value=padding_color)

            if len(self.image_paths) == 1:
                output_path = self.output_img_path + f"/{new_name}" + f".{file_type}"
            elif len(self.image_paths) > 1:
                output_path = self.output_img_path + f"/{new_name}" + f"{index+1}" + f".{file_type}"
            cv2.imwrite(output_path, resized_image)
        print("Done.")


def dataset_loader(image_paths, shape):
    images = np.empty(shape, dtype="uint8") # shape = (400, 128, 128) (number of images, width, height, channels)
    image_labels = []
    for (index, image_path) in enumerate(image_paths):
        print("[INFO] loading image {}/{}".format(index+1, len(image_paths)))
        image = cv2.imread(image_path, 0)
        image = img_to_array(image)
        image_label = image_pths.split(os.path.sep)[-2]
        image_labels.append(image_label)
        images[index] = image
    return (images, np.array(image_labels))