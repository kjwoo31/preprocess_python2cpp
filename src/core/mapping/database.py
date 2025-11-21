"""Mapping Database for Python-to-C++ function mappings."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json


@dataclass
class FunctionMapping:
    """
    Represents a mapping from Python function to C++ implementation.

    Attributes:
        python_lib: Python library name (e.g., 'cv2', 'numpy')
        python_func: Python function name (e.g., 'imread', 'reshape')
        cpp_lib: C++ library name (e.g., 'cv', 'Eigen')
        cpp_func: C++ function/method name
        cpp_header: Required C++ header files
        is_method: Whether this is a method call on an object
        arg_mapping: How to map arguments (optional)
        custom_template: Custom code template for complex mappings
        notes: Additional notes about the mapping
    """
    python_lib: str
    python_func: str
    cpp_lib: str
    cpp_func: str
    cpp_headers: List[str]
    is_method: bool = False
    arg_mapping: Optional[Dict[str, str]] = None
    custom_template: Optional[str] = None
    notes: str = ""


@dataclass
class ConstantMapping:
    """
    Represents a mapping from Python constant to C++ constant.

    Attributes:
        python_lib: Python library name (e.g., 'cv2', 'numpy')
        python_const: Python constant name (e.g., 'IMREAD_GRAYSCALE')
        cpp_value: C++ constant value
        notes: Additional notes about the mapping
    """
    python_lib: str
    python_const: str
    cpp_value: str
    notes: str = ""


class MappingDatabase:
    """
    Database of Python-to-C++ function mappings.

    This class provides a searchable database of known mappings
    and can be extended with custom rules.
    """

    def __init__(self, auto_load_learned: bool = True):
        """
        Initialize mapping database.

        Args:
            auto_load_learned: If True, automatically load learned_mappings.json
        """
        self.mappings: Dict[str, FunctionMapping] = {}
        self.constants: Dict[str, ConstantMapping] = {}
        self._initialize_default_mappings()

        # Auto-load learned mappings if available
        if auto_load_learned:
            self._load_learned_mappings()

    def _initialize_default_mappings(self):
        """Initialize the database with common mappings."""
        self._add_opencv_mappings()
        self._add_opencv_constants()
        self._add_numpy_creation_mappings()
        self._add_numpy_method_mappings()
        self._add_numpy_function_mappings()
        self._add_librosa_mappings()
        self._add_pil_mappings()

    def _add_opencv_mappings(self):
        """Add OpenCV function mappings."""
        # Map to header-only image.h implementation
        self.add_mapping(FunctionMapping(
            python_lib='cv2',
            python_func='imread',
            cpp_lib='img',
            cpp_func='imread',
            cpp_headers=[],
            notes='Reads image from file (handled by image.h)'
        ))

        self.add_mapping(FunctionMapping(
            python_lib='cv2',
            python_func='resize',
            cpp_lib='img',
            cpp_func='resize',
            cpp_headers=[],
            notes='Resizes image (handled by image.h)'
        ))

        self.add_mapping(FunctionMapping(
            python_lib='cv2',
            python_func='cvtColor',
            cpp_lib='img',
            cpp_func='cvtColor_BGR2RGB',
            cpp_headers=[],
            notes='Converts BGR to RGB (handled by image.h)'
        ))

        self.add_mapping(FunctionMapping(
            python_lib='cv2',
            python_func='GaussianBlur',
            cpp_lib='img',
            cpp_func='GaussianBlur',
            cpp_headers=[],
            notes='Applies Gaussian blur (handled by image.h)'
        ))

        self.add_mapping(FunctionMapping(
            python_lib='cv2',
            python_func='bilateralFilter',
            cpp_lib='img',
            cpp_func='bilateralFilter',
            cpp_headers=[],
            notes='Applies bilateral filter (handled by image.h)'
        ))

    def _add_opencv_constants(self):
        """Add OpenCV constant mappings."""
        self.add_constant(ConstantMapping(
            python_lib='cv2',
            python_const='IMREAD_GRAYSCALE',
            cpp_value='0',
            notes='Grayscale image read mode'
        ))

        self.add_constant(ConstantMapping(
            python_lib='cv2',
            python_const='IMREAD_COLOR',
            cpp_value='1',
            notes='Color image read mode'
        ))

        self.add_constant(ConstantMapping(
            python_lib='cv2',
            python_const='IMREAD_UNCHANGED',
            cpp_value='-1',
            notes='Unchanged image read mode'
        ))

        self.add_constant(ConstantMapping(
            python_lib='cv2',
            python_const='COLOR_BGR2RGB',
            cpp_value='COLOR_BGR2RGB',
            notes='BGR to RGB color conversion code'
        ))

        self.add_constant(ConstantMapping(
            python_lib='cv2',
            python_const='COLOR_RGB2BGR',
            cpp_value='COLOR_RGB2BGR',
            notes='RGB to BGR color conversion code'
        ))

        self.add_constant(ConstantMapping(
            python_lib='cv2',
            python_const='COLOR_BGR2GRAY',
            cpp_value='COLOR_BGR2GRAY',
            notes='BGR to grayscale conversion code'
        ))

    def _add_numpy_creation_mappings(self):
        """Add NumPy array creation function mappings."""
        eigen_headers = ['<Eigen/Dense>']

        self.add_mapping(FunctionMapping(
            python_lib='numpy',
            python_func='zeros',
            cpp_lib='Eigen',
            cpp_func='MatrixXf::Zero',
            cpp_headers=eigen_headers,
            custom_template='Eigen::MatrixXf::Zero({rows}, {cols})',
            notes='Creates zero matrix'
        ))

        self.add_mapping(FunctionMapping(
            python_lib='numpy',
            python_func='ones',
            cpp_lib='Eigen',
            cpp_func='MatrixXf::Ones',
            cpp_headers=eigen_headers,
            custom_template='Eigen::MatrixXf::Ones({rows}, {cols})',
            notes='Creates ones matrix'
        ))

        self.add_mapping(FunctionMapping(
            python_lib='numpy',
            python_func='array',
            cpp_lib='Eigen',
            cpp_func='Map',
            cpp_headers=eigen_headers,
            notes='Creates array from data'
        ))

    def _add_numpy_method_mappings(self):
        """Add NumPy array method mappings."""
        method_configs = [
            ('astype', 'img', 'astype_float32', [], 'Converts to float32 (handled by image.h)'),
            ('reshape', 'img', 'reshape', [], 'Reshapes array (handled by image.h)'),
            ('transpose', 'Eigen', 'transpose', ['<Eigen/Dense>'], 'Transposes array'),
        ]

        self._add_mappings_from_config('numpy.ndarray', method_configs, is_method=True)

    def _add_numpy_function_mappings(self):
        """Add NumPy function mappings."""
        self._add_numpy_stat_mappings()
        self._add_numpy_minmax_mappings()

    def _add_numpy_stat_mappings(self):
        """Add NumPy statistical function mappings."""
        self.add_mapping(FunctionMapping(
            python_lib='numpy',
            python_func='mean',
            cpp_lib='cv',
            cpp_func='mean',
            cpp_headers=['<opencv2/opencv.hpp>'],
            notes='Computes mean'
        ))

        self.add_mapping(FunctionMapping(
            python_lib='numpy',
            python_func='std',
            cpp_lib='Eigen',
            cpp_func='mean',
            cpp_headers=['<Eigen/Dense>', '<cmath>'],
            custom_template='std::sqrt(({array} - {array}.mean()).square().mean())',
            notes='Computes standard deviation'
        ))

    def _add_numpy_minmax_mappings(self):
        """Add NumPy min/max function mappings."""
        minmax_mappings = [
            ('argmax', 'img', 'argmax', [], 'Returns index of maximum value'),
            ('max', 'img', 'max', [], 'Returns maximum value'),
            ('argmin', 'img', 'argmin', [], 'Returns index of minimum value'),
            ('min', 'img', 'min', [], 'Returns minimum value'),
        ]

        self._add_mappings_from_config('numpy', minmax_mappings)

    def _add_librosa_mappings(self):
        """Add Librosa function mappings."""
        mappings = [
            ('load', 'sndfile', 'sf_open', ['<sndfile.h>'], 'Loads audio file'),
            ('stft', 'fftw', 'fftw_plan_dft_1d', ['<fftw3.h>'], 'Short-time Fourier transform'),
        ]

        self._add_mappings_from_config('librosa', mappings)

    def _add_mappings_from_config(self, python_lib: str, configs: list,
                                  is_method: bool = False) -> None:
        """Add mappings from configuration list."""
        for python_func, cpp_lib, cpp_func, headers, notes in configs:
            self.add_mapping(FunctionMapping(
                python_lib=python_lib,
                python_func=python_func,
                cpp_lib=cpp_lib,
                cpp_func=cpp_func,
                cpp_headers=headers,
                is_method=is_method,
                notes=notes
            ))

    def _add_pil_mappings(self):
        """Add PIL/Pillow function mappings."""
        self.add_mapping(FunctionMapping(
            python_lib='PIL.Image',
            python_func='open',
            cpp_lib='cv',
            cpp_func='imread',
            cpp_headers=['<opencv2/opencv.hpp>'],
            notes='Opens image file (use OpenCV instead)'
        ))

    def add_mapping(self, mapping: FunctionMapping):
        """
        Add a new mapping to the database.

        Args:
            mapping: FunctionMapping to add
        """
        key = f"{mapping.python_lib}.{mapping.python_func}"
        self.mappings[key] = mapping

    def add_constant(self, constant: ConstantMapping):
        """
        Add a new constant mapping to the database.

        Args:
            constant: ConstantMapping to add
        """
        key = f"{constant.python_lib}.{constant.python_const}"
        self.constants[key] = constant

    def get_constant(self, python_lib: str, python_const: str) -> Optional[ConstantMapping]:
        """
        Get mapping for a Python constant.

        Args:
            python_lib: Python library name (e.g., 'cv2')
            python_const: Python constant name (e.g., 'IMREAD_GRAYSCALE')

        Returns:
            ConstantMapping if found, None otherwise
        """
        key = f"{python_lib}.{python_const}"
        return self.constants.get(key)

    def get_mapping(self, python_lib: str, python_func: str) -> Optional[FunctionMapping]:
        """
        Get mapping for a Python function.

        Args:
            python_lib: Python library name
            python_func: Python function name

        Returns:
            FunctionMapping if found, None otherwise
        """
        key = f"{python_lib}.{python_func}"
        return self.mappings.get(key)

    def get_all_mappings(self) -> List[FunctionMapping]:
        """Get all mappings in the database"""
        return list(self.mappings.values())

    def get_required_headers(self, mappings: List[FunctionMapping]) -> List[str]:
        """
        Get all unique headers required for a list of mappings.

        Args:
            mappings: List of FunctionMapping objects

        Returns:
            List of unique header file includes
        """
        headers = set()
        for mapping in mappings:
            headers.update(mapping.cpp_headers)
        return sorted(list(headers))

    def export_to_json(self, filepath: str):
        """Export mappings to JSON file"""
        data = []
        for mapping in self.mappings.values():
            data.append({
                'python_lib': mapping.python_lib,
                'python_func': mapping.python_func,
                'cpp_lib': mapping.cpp_lib,
                'cpp_func': mapping.cpp_func,
                'cpp_headers': mapping.cpp_headers,
                'is_method': mapping.is_method,
                'notes': mapping.notes
            })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filepath: str):
        """Import mappings from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for item in data:
            mapping = FunctionMapping(
                python_lib=item['python_lib'],
                python_func=item['python_func'],
                cpp_lib=item['cpp_lib'],
                cpp_func=item['cpp_func'],
                cpp_headers=item['cpp_headers'],
                is_method=item.get('is_method', False),
                notes=item.get('notes', '')
            )
            self.add_mapping(mapping)

    def _load_learned_mappings(self):
        """
        Load learned mappings from learned_mappings.json if it exists.

        This allows the system to reuse LLM-generated mappings from previous runs.
        """
        import os
        from pathlib import Path

        # Try to find learned_mappings.json in project root
        project_root = Path(__file__).parent.parent.parent
        learned_file = project_root / 'learned_mappings.json'

        if learned_file.exists():
            try:
                initial_count = len(self.mappings)
                self.import_from_json(str(learned_file))
                num_learned = len(self.mappings) - initial_count
                if num_learned > 0:
                    print(f"✓ Loaded {num_learned} learned mapping(s) from learned_mappings.json")
            except Exception as e:
                print(f"⚠️  Warning: Failed to load learned mappings: {e}")

    def save_learned_mapping(self, mapping: FunctionMapping, learned_file: str = None):
        """
        Save a new learned mapping to the learned_mappings.json file.

        Args:
            mapping: The mapping to save
            learned_file: Path to learned mappings file (default: learned_mappings.json)
        """
        from pathlib import Path

        if learned_file is None:
            project_root = Path(__file__).parent.parent.parent
            learned_file = str(project_root / 'learned_mappings.json')

        # Add to current database
        self.add_mapping(mapping)

        # Load existing learned mappings
        learned_mappings = []
        if Path(learned_file).exists():
            try:
                with open(learned_file, 'r') as f:
                    learned_mappings = json.load(f)
            except:
                learned_mappings = []

        # Check if this mapping already exists
        key = f"{mapping.python_lib}.{mapping.python_func}"
        existing = [m for m in learned_mappings
                   if f"{m['python_lib']}.{m['python_func']}" == key]

        if not existing:
            # Add new mapping
            learned_mappings.append({
                'python_lib': mapping.python_lib,
                'python_func': mapping.python_func,
                'cpp_lib': mapping.cpp_lib,
                'cpp_func': mapping.cpp_func,
                'cpp_headers': mapping.cpp_headers,
                'is_method': mapping.is_method,
                'notes': mapping.notes + ' [LLM-learned]'
            })

            # Save back to file
            with open(learned_file, 'w') as f:
                json.dump(learned_mappings, f, indent=2)

            print(f"✓ Saved learned mapping: {key} → {mapping.cpp_lib}::{mapping.cpp_func}")
