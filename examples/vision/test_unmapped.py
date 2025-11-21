"""Test unmapped operation."""

import cv2


def test_unmapped_func(image_path: str):
    """Test function with unmapped operation."""
    img = cv2.imread(image_path)
    # This function is not in our mapping database
    result = cv2.Canny(img, 100, 200)
    return result
