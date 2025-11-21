"""
MNIST Digit Recognition Pipeline

This demonstrates a complete ML inference pipeline:
- Preprocessing: Load image, resize to 28x28, normalize
- Inference: Model prediction (placeholder for trained model)
- Postprocessing: Get predicted digit class

Usage:
    python src/cli/main.py -i examples/vision/mnist_inference.py --pipeline -o .build/mnist
"""

import cv2
import numpy as np


def mnist_inference(image_path: str) -> int:
    """
    Complete MNIST digit recognition pipeline.

    Args:
        image_path: Path to input image (handwritten digit)

    Returns:
        Predicted digit (0-9)
    """
    # Preprocessing Stage
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (28, 28))
    normalized = resized.astype(np.float32) / 255.0
    flattened = normalized.reshape(1, 784)  # Flatten for model input

    # @inference
    # In production, this would be:
    # predictions = model.predict(flattened)
    # For now, just pass through the normalized image
    predictions = flattened

    # Postprocessing Stage
    # Mock: return predicted digit 3 (hardcoded for demo)
    digit = 3

    return digit


def mnist_with_confidence(image_path: str) -> tuple:
    """
    MNIST inference with confidence score.

    Args:
        image_path: Path to input image

    Returns:
        (predicted_digit, confidence_score)
    """
    # Preprocessing Stage
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (28, 28))
    normalized = resized.astype(np.float32) / 255.0

    # @inference
    # Model prediction (pass through for demo)
    predictions = normalized

    # Postprocessing Stage
    # Mock: return predicted digit 3 with confidence 0.6
    digit = 3
    confidence = 0.6

    return (digit, confidence)


if __name__ == '__main__':
    # Test with a sample image
    result = mnist_inference("test_digit.jpg")
    print(f"Predicted digit: {result}")

    digit, conf = mnist_with_confidence("test_digit.jpg")
    print(f"Digit: {digit}, Confidence: {conf:.2%}")
