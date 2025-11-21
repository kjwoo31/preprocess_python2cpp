"""Advanced preprocessing with unmapped operations."""

import cv2
import numpy as np


def denoise_image(image_path: str):
    """
    Denoise image using bilateral filter (unmapped operation).

    Args:
        image_path: Path to input image

    Returns:
        Denoised image
    """
    img = cv2.imread(image_path)
    # bilateralFilter is not in default mappings - will use LLM
    denoised = cv2.bilateralFilter(img, 9, 75, 75)
    return denoised
