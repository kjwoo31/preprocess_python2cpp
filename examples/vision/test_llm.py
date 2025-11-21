"""Test LLM generation with unmapped operation."""

import cv2


def apply_morph(image_path: str):
    """
    Apply morphological operation (unmapped).

    Args:
        image_path: Path to input image

    Returns:
        Processed image
    """
    img = cv2.imread(image_path)
    # morphologyEx is not mapped - will use LLM
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return result
