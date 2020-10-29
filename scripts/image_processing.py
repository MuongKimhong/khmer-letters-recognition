import os
import cv2
from imutils import paths
import numpy as np

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
        print("Done")


'''
How to use ResizeImage:

1. For input_image_path, if image's path is "path/to/image/my_name.jpg" => input_image_paths = "path/to/image"

2. For output_image_path, if output path is "path/to/output/new_name.jpg" => output_image_paths = "path/to/output"

3. size is tuple of new height and new width, Example, (64, 64), (200, 200), ..

4. For new_name, if rename only 1 image then new file name will be the same as new_name provided but if rename more than
   one image, the new file names will be new_name1.jpg, new_name2.jpg, new_name3.jpg, ...
'''
class ResizeImage:
    def __init__(self, input_image_paths, output_image_paths, size, new_name, file_type="jpg", padding_color=0):
        self.input_image_paths = input_image_paths
        self.output_image_paths = output_image_paths
        self.padding_color = padding_color
        self.new_name = new_name
        self.file_type = file_type
        self.size = size

    def crop(self, image, top, bottom, left, right):
        image_height, image_width = image.shape[:2]
        cropped_image = image[top:(image_height - bottom), left:(image_width - right)]
        return cropped_image
        
    def resize(self):
        image_paths = list(paths.list_images(self.input_image_paths))
        for (i, image_path) in enumerate(image_paths):
            padding_left, padding_right, padding_top, padding_bottom = 0, 0, 0, 0
            print("[INFO] processing image {} ....".format(i+1))
            image_path = cv2.imread(image_path)
            height, width = image_path.shape[:2]
            new_height, new_width = self.size
            print("[INFO] cropping image {} ....".format(i+1))
            cropped_image = self.crop(image_path, 500, 500, 200, 200)
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
            scaled_image = cv2.copyMakeBorder(scaled_image, padding_top, padding_bottom, padding_left, 
                                              padding_right, borderType=cv2.BORDER_CONSTANT, value=self.padding_color)
            if len(image_paths) == 1:
                output_paths = self.output_image_paths + f"/{self.new_name}" + f".{self.file_type}"
            elif len(image_paths) > 1:
                output_paths = self.output_image_paths + f"/{self.new_name}" + f"{i+1}" + f".{self.file_type}"
            cv2.imwrite(output_paths, scaled_image)
        print("Done.")