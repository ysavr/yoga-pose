import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pathlib import Path
import logging


class ImagePreprocessor:
    def __init__(self, input_dir, output_dir):
        """
        Initialize the ImagePreprocessor.

        Args:
            input_dir (str): Input directory containing category folders with images
            output_dir (str): Output directory for processed images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = (224, 224)
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def create_directories(self):
        """Create output directories if they don't exist"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            self.logger.info(f"Created output directory: {self.output_dir}")

    def is_blurry(self, image, threshold=100):
        """
        Check if image is blurry using Laplacian variance.

        Args:
            image: numpy array of the image
            threshold: blur threshold (lower means more strict)

        Returns:
            bool: True if image is blurry
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def has_text(self, image, text_threshold=50):
        """
        Check if image has significant text using pytesseract.

        Args:
            image: numpy array of the image
            text_threshold: minimum confidence for text detection

        Returns:
            bool: True if significant text is detected
        """
        text = pytesseract.image_to_string(image)
        return len(text.strip()) > text_threshold

    def has_multiple_objects(self, image, threshold=0.6):
        """
        Detect if image has multiple distinct objects using contour detection.

        Args:
            image: numpy array of the image
            threshold: threshold for object detection

        Returns:
            bool: True if multiple objects detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        significant_contours = 0
        image_area = image.shape[0] * image.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > image_area * threshold:
                significant_contours += 1

        return significant_contours > 1

    def has_artifacts(self, image, threshold=10):
        """
        Detect digital artifacts using edge detection variance.

        Args:
            image: numpy array of the image
            threshold: artifact detection threshold

        Returns:
            bool: True if artifacts detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_variance = np.var(edges)
        return edge_variance > threshold

    def normalize_image(self, image):
        """
        Normalize image brightness and contrast.

        Args:
            image: numpy array of the image

        Returns:
            normalized image
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        normalized = cv2.merge((cl, a, b))
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

    def remove_noise(self, image):
        """
        Remove noise from image using bilateral filter.

        Args:
            image: numpy array of the image
            Returns:
            denoised image
        """
        return cv2.bilateralFilter(image, 9, 75, 75)

    def resize_with_padding(self, image):
        """
        Resize image to target size with padding to maintain aspect ratio.

        Args:
            image: numpy array of the image

        Returns:
            resized image
        """
        h, w = image.shape[:2]
        target_size = self.target_size[0]

        # Calculate scaling factor to fit within target size
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h))

        # Create square canvas
        square = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Calculate padding
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2

        # Place image in center
        square[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return square

    def process_image(self, image_path):
        """
        Process a single image according to specifications.

        Args:
            image_path: Path to the image file

        Returns:
            bool: True if image was processed successfully
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Could not read image: {image_path}")
                return False

            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Check quality criteria
            if self.is_blurry(image):
                self.logger.info(f"Skipping blurry image: {image_path}")
                return False

            if self.has_text(image):
                self.logger.info(f"Skipping image with text: {image_path}")
                return False

            if self.has_multiple_objects(image):
                self.logger.info(f"Skipping image with multiple objects: {image_path}")
                return False

            # if self.has_artifacts(image):
            #     self.logger.info(f"Skipping image with artifacts: {image_path}")
            #     return False

            # Process image
            image = self.normalize_image(image)
            image = self.remove_noise(image)
            image = self.resize_with_padding(image)

            # Convert back to BGR for saving
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            return image

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return False

    def process_directory(self):
        """Process all images in the input directory structure"""
        self.create_directories()

        for category_dir in self.input_dir.iterdir():
            if category_dir.is_dir():
                # Create category directory in output
                output_category_dir = self.output_dir / category_dir.name
                output_category_dir.mkdir(exist_ok=True)

                self.logger.info(f"Processing category: {category_dir.name}")

                for image_path in category_dir.glob("*.[jJ][pP][gG]"):
                    processed_image = self.process_image(image_path)

                    if processed_image is not False:
                        # Save as PNG
                        output_path = output_category_dir / f"{image_path.stem}.png"
                        cv2.imwrite(str(output_path), processed_image)
                        self.logger.info(f"Processed and saved: {output_path}")


def main():
    # Example usage
    processor = ImagePreprocessor(
        input_dir="data/images",
        output_dir="data/images_processed",
    )
    processor.process_directory()


if __name__ == "__main__":
    main()