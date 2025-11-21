"""Simple image loading example."""

import cv2


def load_image(image_path: str):
    """
    Load an image from file.

    Args:
        image_path: Path to input image

    Returns:
        Loaded image
    """
    img = cv2.imread(image_path)
    return img
