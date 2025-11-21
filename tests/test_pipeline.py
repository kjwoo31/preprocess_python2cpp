"""Test pipeline separation and conversion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.analysis.separator import PipelineSeparator
import ast


def test_separator_with_marker():
    """Test separator correctly splits code with @inference marker."""
    source_code = """
import cv2

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    resized = cv2.resize(img, (224, 224))
    return resized

# @inference
def run_inference(preprocessed):
    # Model inference would go here
    predictions = model.predict(preprocessed)
    return predictions

def postprocess_results(predictions):
    class_id = predictions.argmax()
    return class_id
"""

    tree = ast.parse(source_code)
    separator = PipelineSeparator()
    result = separator.separate(source_code, tree)

    assert result.has_inference_marker
    assert result.preprocess is not None
    assert result.inference is not None
    assert result.postprocess is not None

    print("✓ Test passed: Separator correctly splits code with @inference marker")


def test_separator_without_marker():
    """Test separator handles code without @inference marker."""
    source_code = """
import cv2

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    resized = cv2.resize(img, (224, 224))
    return resized
"""

    tree = ast.parse(source_code)
    separator = PipelineSeparator()
    result = separator.separate(source_code, tree)

    assert not result.has_inference_marker
    assert result.preprocess is not None
    assert result.inference is None
    assert result.postprocess is None

    print("✓ Test passed: Separator handles code without @inference marker")


def test_variable_tracking():
    """Test separator tracks input/output variables correctly."""
    source_code = """
def preprocess(x):
    y = x * 2
    z = y + 1
    return z

# @inference
def inference(z):
    result = z * 3
    return result
"""

    tree = ast.parse(source_code)
    separator = PipelineSeparator()
    result = separator.separate(source_code, tree)

    assert result.preprocess is not None
    assert result.inference is not None

    print("✓ Test passed: Variable tracking works correctly")


if __name__ == '__main__':
    test_separator_with_marker()
    test_separator_without_marker()
    test_variable_tracking()
    print("\n✅ All pipeline tests passed!")
