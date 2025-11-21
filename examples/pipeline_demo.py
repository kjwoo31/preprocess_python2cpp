"""
Pipeline Demo: Image Classification Pipeline with Separated Stages

This example demonstrates the @inference marker for pipeline separation:
- Preprocessing: Image resize and normalization
- Inference: Model prediction (placeholder)
- Postprocessing: Class label lookup

Usage:
    python src/cli/main.py -i examples/pipeline_demo.py --pipeline -o .build/pipeline_demo
"""

import cv2
import numpy as np


def pipeline_demo(image_path: str) -> str:
    """
    Complete image classification pipeline.

    Args:
        image_path: Path to input image

    Returns:
        Predicted class label
    """
    # Preprocessing Stage
    img = cv2.imread(image_path)
    resized = cv2.resize(img, (224, 224))
    normalized = resized.astype(np.float32) / 255.0

    # @inference
    # Inference Stage - Model prediction happens here
    # In production, this would call a trained model
    predictions = np.array([0.1, 0.2, 0.7])  # Mock predictions

    # Postprocessing Stage
    class_id = int(np.argmax(predictions))
    class_labels = ["cat", "dog", "bird"]
    result = class_labels[class_id]

    return result


if __name__ == '__main__':
    result = pipeline_demo("test_image.jpg")
    print(f"Prediction: {result}")
