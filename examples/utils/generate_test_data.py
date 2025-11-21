"""
Generate Test Data

Utility script to create test images for validation.
"""

import numpy as np
import cv2
import os


def create_test_image(output_path: str = "test_image.jpg", size: int = 512):
    """
    Create a simple test image with gradient pattern.

    Args:
        output_path: Path to save the test image
        size: Image size (square)
    """
    # Create gradient image
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for i in range(size):
        for j in range(size):
            img[i, j, 0] = int((i / size) * 255)  # Red channel
            img[i, j, 1] = int((j / size) * 255)  # Green channel
            img[i, j, 2] = int(((i + j) / (2 * size)) * 255)  # Blue channel

    # Save image
    cv2.imwrite(output_path, img)
    print(f"✓ Created test image: {output_path} ({size}x{size})")


if __name__ == '__main__':
    # Create test image in config/data/
    os.makedirs('config/data', exist_ok=True)
    create_test_image('config/data/test_image.jpg', size=512)
