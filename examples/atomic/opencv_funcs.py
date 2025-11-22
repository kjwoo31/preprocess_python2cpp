"""
Atomic OpenCV Function Tests

Individual OpenCV functions for granular testing.
"""

import cv2
import numpy as np


def test_imread(image_path: str):
    """Test cv2.imread only."""
    img = cv2.imread(image_path)
    return img


def test_imread_grayscale(image_path: str):
    """Test cv2.imread with grayscale flag."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


def test_resize(image_path: str):
    """Test cv2.resize only."""
    img = cv2.imread(image_path)
    resized = cv2.resize(img, (224, 224))
    return resized


def test_cvtColor(image_path: str):
    """Test cv2.cvtColor BGR to RGB."""
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb


def test_cvtColor_gray(image_path: str):
    """Test cv2.cvtColor BGR to GRAY."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def test_GaussianBlur(image_path: str):
    """Test cv2.GaussianBlur."""
    img = cv2.imread(image_path)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return blurred


def test_threshold(image_path: str):
    """Test cv2.threshold."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return thresh


def test_Canny(image_path: str):
    """Test cv2.Canny edge detection."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    return edges
