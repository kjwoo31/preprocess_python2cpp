"""
Image Preprocessing Example

Simple image preprocessing functions demonstrating:
- Image loading and resizing
- Type conversion and normalization
- Color space conversion
"""

import cv2
import numpy as np
import sys


def preprocess_image(image_path: str):
    """
    Basic image preprocessing: resize and normalize.

    Args:
        image_path: Path to input image

    Returns:
        Preprocessed image as float32 array
    """
    # Load image
    img = cv2.imread(image_path)

    # Resize to 224x224
    img = cv2.resize(img, (224, 224))

    # Convert to float and normalize
    img = img.astype(np.float32) / 255.0

    return img


def preprocess_with_color_conversion(image_path: str):
    """
    Image preprocessing with BGR to RGB conversion.

    Args:
        image_path: Path to input image

    Returns:
        Preprocessed image with RGB color space
    """
    # Load image (OpenCV loads as BGR)
    img = cv2.imread(image_path)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 256x256
    img = cv2.resize(img, (256, 256))

    # Convert to float
    img_float = img.astype(np.float32)

    return img_float


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python image_preprocessing.py <image_path>")
        print("Example: python image_preprocessing.py test.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Processing image: {image_path}")

    # Test preprocess_image
    result1 = preprocess_image(image_path)
    print(f"preprocess_image: shape={result1.shape}, dtype={result1.dtype}")

    # Test preprocess_with_color_conversion
    result2 = preprocess_with_color_conversion(image_path)
    print(f"preprocess_with_color_conversion: shape={result2.shape}, dtype={result2.dtype}")
