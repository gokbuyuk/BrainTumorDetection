import os
import cv2
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from src.utils.utils import *

print('working directory is:')
print(os.getcwd())
# Replace with your actual paths
INPUT_TRAIN_PATH = 'data/raw/Training'
INPUT_TEST_PATH = 'data/raw/Testing'
# OUTPUT_TRAIN_PATH = 'data/interim/Training'
# OUTPUT_TEST_PATH = 'data/interim/Testing'

# for input_path in [INPUT_TRAIN_PATH, INPUT_TEST_PATH]:
#     for category in os.listdir(input_path):
#         if category == '.DS_Store':
#             continue
#         category_path = os.path.join(input_path, category)
#         ic(category_path)
#         # Iterate over each image in the category
#         for image_name in os.listdir(category_path):
#             image_path = os.path.join(category_path, image_name)
#             # ic(image_path)
#             image = cv2.imread(image_path)

#             # crop the image and save it
#             cropped_image = crop_black_frame(image_path)
#             if cropped_image is None:
#                 continue
#             # save it to the output directory with the same subdirectory structure
#             output_path = os.path.join(input_path.replace('raw', 'interim/cropped'), category)
#             os.makedirs(output_path, exist_ok=True)
#             output_image_path = os.path.join(output_path, image_name)
#             cv2.imwrite(output_image_path, cropped_image)


crop_black_frame_in_directory(INPUT_TRAIN_PATH, INPUT_TRAIN_PATH.replace('raw', 'interim/cropped'))

downsample_images('data/interim/cropped', 'data/interim/resized')
