"""
Atomic Type Conversion Tests

Type conversion and casting operations.
"""

import cv2
import numpy as np


def test_int_conversion(image_path: str):
    """Test int() conversion."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean_val = np.mean(img)
    result = int(mean_val)
    return result


def test_float_conversion(image_path: str):
    """Test float() conversion."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean_val = np.mean(img)
    result = float(mean_val)
    return result


def test_bool_operation(image_path: str):
    """Test boolean comparison."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    result = img > 128
    return result.astype(np.uint8)


def test_tuple_return(image_path: str):
    """Test tuple return."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height = img.shape[0]
    width = img.shape[1]
    return (height, width)


def test_max_value(image_path: str):
    """Test max value."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    result = int(np.max(img))
    return result


def test_min_value(image_path: str):
    """Test min value."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    result = int(np.min(img))
    return result
