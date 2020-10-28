import os
from imutils import paths


'''
How to use:

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
            os.rename(image_path, output_path)