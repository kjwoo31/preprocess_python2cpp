"""Simple pipeline demonstration with Pre/Inf/Post stages."""

import cv2
import numpy as np


def simple_pipeline(image_path: str):
    """
    Simple preprocessing pipeline demonstration.

    Args:
        image_path: Path to input image

    Returns:
        Processed image
    """
    # Preprocessing
    img = cv2.imread(image_path)
    resized = cv2.resize(img, (224, 224))

    # @inference
    # Placeholder: in production this would be model inference
    result = resized

    # Postprocessing  
    normalized = result.astype(np.float32) / 255.0

    return normalized
