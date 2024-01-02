import cv2
import os

class Preprocessor:
    def __init__(self, target_size=(120, 120)):
        self.target_size = target_size

    def crop_black_frame(self, image_path):
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

    def process_image(self, image_path):
        # Combine cropping and resizing
        cropped_image = self.crop_black_frame(image_path)
        if cropped_image is not None:
            resized_image = cv2.resize(cropped_image, self.target_size, interpolation=cv2.INTER_AREA)
            return resized_image
        return None

    def process_directory(self, input_directory, output_directory):
        for subdir, _, files in os.walk(input_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(subdir, file)
                    processed_image = self.process_image(image_path)

                    if processed_image is not None:
                        relative_path = os.path.relpath(subdir, input_directory)
                        output_subdir = os.path.join(output_directory, relative_path)
                        os.makedirs(output_subdir, exist_ok=True)

                        output_image_path = os.path.join(output_subdir, file)
                        cv2.imwrite(output_image_path, processed_image)


    def get_image_array(self, image_path):
        """
        Reads an image from the given path and returns it as an array.

        Args:
            image_path (str): Path to the image.

        Returns:
            numpy.ndarray: The image array.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
        return image

# Example usage
# preprocessor = Preprocessor()
# preprocessor.process_directory('path/to/input/directory', 'path/to/output/directory')
if __name__ == '__main__':
    INPUT_PATH = 'data/raw'
    preprocessor = Preprocessor()
    # preprocessor.process_directory(INPUT_PATH, INPUT_PATH.replace('raw', 'interim/resized'))
    array = preprocessor.get_image_array('data/interim/resized/Training/meningioma/Tr-me_0010.jpg')
    print(array.shape)
    print(array)
    print(array.max(), array. min(), array.mean())

