import os
import cv2
import numpy as np
# from icecream import ic


def crop_black_frame(image_path):
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


def crop_black_frame_in_directory(input_directory, output_directory):
    """
    Processes all images in the input directory and its subdirectories, crops them using the given function,
    and saves them in the output directory, maintaining the subdirectory structure.

    Args:
    input_directory (str): The directory containing the images to be processed.
    output_directory (str): The directory where the processed images will be saved.
    """
    for subdir, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.'):
                image_path = os.path.join(subdir, file)
                cropped_image = crop_black_frame(image_path)

                if cropped_image is not None:
                    relative_path = os.path.relpath(subdir, input_directory)
                    output_subdir = os.path.join(output_directory, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)

                    output_image_path = os.path.join(output_subdir, file)
                    cv2.imwrite(output_image_path, cropped_image)

# Example usage
# Define your cropping function, e.g., crop_black_frame
# process_images('path/to/input/directory', 'path/to/output/directory', crop_black_frame)



def downsample_images(input_directory, output_directory, target_size=(100, 100)):
    """
    Downsamples all images in the input directory and its subdirectories to the target size
    and saves them in the corresponding subdirectories of the output directory.

    Args:
    input_directory (str): The directory containing the images to be resized.
    output_directory (str): The directory where the resized images will be saved.
    target_size (tuple): The target size for resizing, default is (100, 100).
    """
    for subdir, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, file)
                image = cv2.imread(image_path)

                if image is not None:
                    resized_image = cv2.resize(image, target_size)

                    relative_path = os.path.relpath(subdir, input_directory)
                    output_subdir = os.path.join(output_directory, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)

                    save_path = os.path.join(output_subdir, file)
                    cv2.imwrite(save_path, resized_image)
                else:
                    print(f"Failed to read image: {image_path}")

if __name__ == '__main__':

    # Replace with your actual paths
    INPUT_TRAIN_PATH = 'data/raw/Training'
    INPUT_TEST_PATH = 'data/raw/Testing'
    crop_black_frame_in_directory(INPUT_TRAIN_PATH, INPUT_TRAIN_PATH.replace('raw', 'interim/cropped'))

    downsample_images('data/interim/cropped', 'data/interim/resized')