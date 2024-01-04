import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
from PIL import Image

def find_smallest_dimensions(directory):
    smallest_width = float('inf')
    smallest_height = float('inf')

    for subdir, _, files in os.walk(directory):
        # ic(subdir)
        for filename in files:
            if filename.endswith(".jpg"):
                image_path = os.path.join(subdir, filename)
                with Image.open(image_path) as img:
                    width, height = img.size
                    smallest_width = min(smallest_width, width)
                    smallest_height = min(smallest_height, height)

    return smallest_width, smallest_height

def get_metadata(path, data_type):
    """
    Extracts metadata from images in a specified directory and returns a DataFrame.

    Args:
        path (str): The file path to the directory containing subdirectories of images.
        data_type (str): A label representing the type of data (e.g., 'train', 'test').

    Returns:
        pd.DataFrame: A DataFrame containing the metadata of the images including
                      paths, sizes, areas, labels, and data types.
    """
    data = []

    # Iterate over each category (subdirectory)
    for category in os.listdir(path):
        if category == '.DS_Store':
            continue
        category_path = os.path.join(path, category)
        # ic(category_path)
        # Iterate over each image in the category
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            # ic(image_path)
            image = cv2.imread(image_path)

            # Skip if image is not found
            if image is None:
                continue

            size = image.shape
            area = size[0] * size[1]

            # Append metadata to the list
            data.append({
                'Path': image_path,
                'Data': data_type,
                'Label': category,
                'Size': size,
                'Area': area
            })

    return pd.DataFrame(data)


def display_image(image_path, title, ax):
    """
    Display an image using Matplotlib.

    This function reads an image from the specified file path, and then displays it 
    on a given Matplotlib Axes object. The image is displayed with the provided title 
    and with axis lines turned off.

    Args:
        image_path (str): Path to the image file to be displayed.
        title (str): Title to be set for the subplot.
        ax (matplotlib.axes.Axes): The Axes object in which the image will be displayed.

    Returns:
        None
    """
    image = cv2.imread(image_path)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')