# standard library import
import os
from dataclasses import dataclass
# third party umport
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import cv2


'''
How to use RenameFile:

1. For input_image_path, if image's path is "path/to/image/my_name.jpg" => input_image_path = "path/to/image"

2. For output_image_path, if output path is "path/to/output/new_name.jpg" => output_image_path = "path/to/output"

3. For new_name, if rename only 1 image then new file name will be the same as new_name provided but if rename more than
   one image, the new file names will be new_name1.jpg, new_name2.jpg, new_name3.jpg, ...

4. If not provide, output file name will be new_name.jpg, option is "jpg" or "png" 
'''
class RenameFile:
    def __init__(self, input_image_path, output_image_path, new_name, file_type="jpg"):
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.new_name = new_name
        self.file_type = file_type

    def rename(self):
        image_paths = list(paths.list_images(self.input_image_path))
        for (i, image_path) in enumerate(image_paths):
            if len(image_paths) == 1:
                output_path = self.output_image_path + "/" + f"{self.new_name}" + f".{self.file_type}"
            elif len(image_paths) > 1:
                output_path = self.output_image_path + "/" + f"{self.new_name}" + f"{i+1}" + f".{self.file_type}"
            print("[INFO] renaming image {} ....".format(i+1))
            os.rename(image_path, output_path)
        print("Done.")


'''
How to use ResizeImage:

1. For input_image_path, if image's path is "path/to/image/my_name.jpg" => input_image_paths = "path/to/image"

2. For output_image_path, if output path is "path/to/output/new_name.jpg" => output_image_paths = "path/to/output"

3. size is tuple of new height and new width, Example, (64, 64), (200, 200), ..

4. For new_name, if rename only 1 image then new file name will be the same as new_name provided but if rename more than
   one image, the new file names will be new_name1.jpg, new_name2.jpg, new_name3.jpg, ...

5. if crop_image = True, need to provide values for top, bottom, left, right, default is False
'''
@dataclass(init=True)
class ResizeImage:
    input_image_paths: str
    output_image_paths: str
    size: list
    new_name: str
    crop_image: bool = False
    top: int = None
    bottom: int = None
    left: int = None
    right: int = None
    file_type: str = "jpg"
    padding_color: int = 0

    def crop(self, image):
        image_height, image_width = image.shape[:2]
        layout = {'top': self.top, 'bottom': self.bottom, 'left': self.left, 'right': self.right}
        cropped_image = image[layout['top']:(image_height - layout['bottom']), layout['left']:(image_width - layout['right'])]
        return cropped_image

    def rotate(self, image, angle):
        if angle == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        return rotated_image

    def flip(self, image):
        flipped_image = cv2.flip(image, 1)
        return flipped_image
        
    def resize(self):
        image_paths = list(paths.list_images(self.input_image_paths))
        for (i, image_path) in enumerate(image_paths):
            padding = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
            print("[INFO] processing image {} ....".format(i+1))
            image_path = cv2.imread(image_path)
            height, width = image_path.shape[:2]
            new_height, new_width = self.size
            if self.crop_image and ((self.top is None) or (self.bottom is None) or (self.left is None) or (self.right is None)):
                error = "crop_image is True but top or bottom or left or right value is not provided"
                exeption = print(error)
                return exeption
            if self.crop_image and (self.top is not None) and (self.bottom is not None) and (self.left is not None) and (self.right is not None):
                print("[INFO] cropping image {} ....".format(i+1))
                cropped_image = self.crop(image_path)
                cropped_image_height, cropped_image_width = cropped_image.shape[:2]
                # Interpolation method
                if (cropped_image_height > new_height) or (cropped_image_width > new_width):
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC
                # Aspect ratio
                aspect_ratio = cropped_image_width / cropped_image_height
                # set padding color 
                if len(cropped_image.shape) == 3 and not isinstance(self.padding_color, (list, tuple, np.ndarray)):
                    self.padding_color = [self.padding_color] * 3
                # scale and pad
                print("[INFO] resizing image {} ....".format(i+1))
                scaled_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=interpolation)
                scaled_image = cv2.copyMakeBorder(scaled_image, padding['top'], padding['bottom'], padding['left'], 
                                                  padding['right'], borderType=cv2.BORDER_CONSTANT, value=self.padding_color)
            else:
                # Interpolation method
                if (height > new_height) or (width > new_width):
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC
                # Aspect ratio
                aspect_ratio = width / height
                # set padding color
                if len(image_path.shape) == 3 and not isinstance(self.padding_color, (list, tuple, np.ndarray)):
                    self.padding_color = [self.padding_color] * 3
                # scale and pad
                print("[INFO] resizing image {} ....".format(i+1))
                scaled_image = cv2.resize(image_path, (new_width, new_height), interpolation=interpolation)
                scaled_image = cv2.copyMakeBorder(scaled_image, padding['top'], padding['bottom'], padding['left'],
                                                  padding['right'], borderType=cv2.BORDER_CONSTANT, value=self.padding_color)
            if len(image_paths) == 1:
                output_paths = self.output_image_paths + f"/{self.new_name}" + f".{self.file_type}"
            elif len(image_paths) > 1:
                output_paths = self.output_image_paths + f"/{self.new_name}" + f"{i+1}" + f".{self.file_type}"
            cv2.imwrite(output_paths, scaled_image)
        print("Done.")


def convert_to_array(image, data_format=None):
    image = image
    data_format = data_format
    array_image = img_to_array(image, data_format=data_format)
    return array_image


class DatasetLoader:
    def __init__(self, image_paths, verbose=-1):
        self.image_paths = image_paths
        self.verbose = verbose
        self.image_labels = []
        self.images = []

    def load(self):
        images = self.images
        image_labels = self.image_labels
        image_paths = self.image_paths
        for (i, image_path) in enumerate(image_paths):
            # /path/to/dataset/class/image.jpg
            image = cv2.imread(image_path)
            image = convert_to_array(image)
            image_label = image_path.split(os.path.sep)[-2]
            image_labels.append(image_label)
            images.append(image)
            if self.verbose > 0 and i > 0 and (i + 1) % self.verbose == 0:
                print("[INFO] loading image {}/{}".format(i + 1, len(image_paths)))
        return (np.array(images), np.array(image_labels))