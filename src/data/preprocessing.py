import os
import cv2
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt


def crop_brain_contour(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None

    _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in image at {image_path}")
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def crop_brain_contour2(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None

    # Adjust the threshold value here. 
    # Lower values will include more lighter areas.
    threshold_value = 30  # Example value, adjust as needed based on your images

    _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in image at {image_path}")
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image


# Replace with your actual paths
INPUT_TRAIN_PATH = 'data/raw/Training'
INPUT_TEST_PATH = 'data/raw/Testing'
# OUTPUT_TRAIN_PATH = 'data/interim/Training'
# OUTPUT_TEST_PATH = 'data/interim/Testing'

for input_path in [INPUT_TRAIN_PATH, INPUT_TEST_PATH]:
    for category in os.listdir(input_path):
        if category == '.DS_Store':
            continue
        category_path = os.path.join(input_path, category)
        ic(category_path)
        # Iterate over each image in the category
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            # ic(image_path)
            image = cv2.imread(image_path)

            # crop the image and save it
            cropped_image = crop_brain_contour2(image_path)
            if cropped_image is None:
                continue
            # save it to the output directory with the same subdirectory structure
            output_path = os.path.join(input_path.replace('raw', 'interim'), category)
            os.makedirs(output_path, exist_ok=True)
            output_image_path = os.path.join(output_path, image_name)
            cv2.imwrite(output_image_path, cropped_image)





