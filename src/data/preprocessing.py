from PIL import Image
import os

def crop_and_resize_images(input_folder, output_folder, target_size):
    #if not os.path.exists(output_folder):
        #os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image
            img = Image.open(input_path)

            # Crop black frames
            img_data = list(img.getdata())
            non_black_pixels = [pixel for pixel in img_data if pixel != (0, 0, 0)]
            if non_black_pixels:
                left, top, right, bottom = img.getbbox()
                img = img.crop((left, top, right, bottom))

                # Resize to the target size
                img = img.resize(target_size)

                # Save the processed image
                img.save(output_path)
                print(f"Processed: {filename}")
            else:
                print(f"Skipped (no content): {filename}")

if __name__ == "__main__":
    # Set the input and output folders
    input_folder = "../data/raw/Testing"
    output_folder = "../data/interim/Testing"

    # Set the target size for the output images
    target_size = (256, 256)  # Replace with your desired size

    crop_and_resize_images(input_folder, output_folder, target_size)

