"""
Atomic NumPy Function Tests

Individual NumPy functions for granular testing.
"""

import cv2
import numpy as np


def test_astype_float32(image_path: str):
    """Test ndarray.astype(np.float32)."""
    img = cv2.imread(image_path)
    result = img.astype(np.float32)
    return result


def test_astype_uint8(image_path: str):
    """Test ndarray.astype(np.uint8)."""
    img = cv2.imread(image_path)
    float_img = img.astype(np.float32) / 255.0
    result = (float_img * 255.0).astype(np.uint8)
    return result


def test_divide_scalar(image_path: str):
    """Test array division by scalar."""
    img = cv2.imread(image_path)
    result = img.astype(np.float32) / 255.0
    return result


def test_multiply_scalar(image_path: str):
    """Test array multiplication by scalar."""
    img = cv2.imread(image_path)
    normalized = img.astype(np.float32) / 255.0
    result = normalized * 2.0
    return result


def test_reshape(image_path: str):
    """Test array reshape."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (28, 28))
    result = resized.reshape(1, 784)
    return result


def test_transpose(image_path: str):
    """Test array transpose."""
    img = cv2.imread(image_path)
    result = img.transpose(2, 0, 1)
    return result


def test_mean(image_path: str):
    """Test np.mean."""
    img = cv2.imread(image_path)
    result = np.mean(img, axis=(0, 1))
    return result


def test_std(image_path: str):
    """Test np.std."""
    img = cv2.imread(image_path)
    result = np.std(img, axis=(0, 1))
    return result


def test_clip(image_path: str):
    """Test np.clip."""
    img = cv2.imread(image_path).astype(np.float32)
    result = np.clip(img, 50, 200)
    return result


def test_subtract(image_path: str):
    """Test array subtraction."""
    img = cv2.imread(image_path).astype(np.float32)
    result = img - 128.0
    return result


def test_add(image_path: str):
    """Test array addition."""
    img = cv2.imread(image_path).astype(np.float32)
    result = img + 10.0
    return result
