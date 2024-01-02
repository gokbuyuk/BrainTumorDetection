from PIL import Image
import os

def find_smallest_dimensions(directory,data,label):
    smallest_width = float('inf')
    smallest_height = float('inf')

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                smallest_width = min(smallest_width, width)
                smallest_height = min(smallest_height, height)

    return smallest_width, smallest_height

# Replace 'your_directory' with the path to the directory containing your .jpg images
dict= ["glioma","meningioma","notumor","pituitary"]
for label in dict:

directory_path = 'data/interim/'+'data'+'/'+'label'

smallest_width, smallest_height = find_smallest_dimensions(directory_path)
print(f"The smallest width of .jpg images is: {smallest_width} pixels")
print(f"The smallest height of .jpg images is: {smallest_height} pixels")
