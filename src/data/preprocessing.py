import os
import cv2
import numpy as np

# Paths to the datasets and outputs
# Directory to save the images
IMAGE_DIR = "../reports/visualizations"
os.makedirs(IMAGE_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# Paths to the datasets
INPUT_TRAIN_PATH = ('../data/raw/Training')
INPUT_TEST_PATH = ('../data/raw/Testing')

OUTPUT_TRAIN_PATH = ('../data/interim/Training')
OUTPUT_TEST_PATH = ('../data/interim/Testing')

# Create output directories if they don't exist
os.makedirs(OUTPUT_TRAIN_PATH, exist_ok=True)
os.makedirs(OUTPUT_TEST_PATH, exist_ok=True)


def crop_brain_contour(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if image is loaded properly
    if image is None:
        print(f"Error: Image not found at {image_path}")
        return None

    # Apply a binary threshold to the image
    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return original image
    if not contours:
        print(f"No contours found in image at {image_path}")
        return image

    # Find the largest contour which will be the contour of the brain
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the dimensions of the bounding rectangle
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image

def crop_images_in_folder(source_folder, output_folder):
    # Iterate over all directories in the source folder
    for subdir, dirs, files in os.walk(source_folder):
        for file in files:
            # Construct the path to the image file
            image_path = os.path.join(subdir, file)
            
            # Check for image formats here (e.g., '.jpg')
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Crop the brain contour from the image
                cropped_image = crop_brain_contour(image_path)

                if cropped_image is not None:
                    # Replicate the subdirectory structure in the output folder
                    relative_path = os.path.relpath(subdir, source_folder)
                    output_subdir = os.path.join(output_folder, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)

                    # Construct the save path and save the cropped image
                    save_path = os.path.join(output_subdir, file)
                    cv2.imwrite(save_path, cropped_image)


crop_images_in_folder(INPUT_TRAIN_PATH, OUTPUT_TRAIN_PATH)
