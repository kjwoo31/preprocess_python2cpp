"""Result Comparator for numerical output validation."""

from pathlib import Path
from typing import Any

import numpy as np


class ResultComparator:
    """
    Compares results from Python and C++ implementations.
    """

    def __init__(self, results_dir: str = ".build/results"):
        """
        Initialize comparator.

        Args:
            results_dir: Directory containing result files
        """
        self.results_dir = Path(results_dir)

    def compare_outputs(
        self, function_name: str, rtol: float = 1e-4, atol: float = 1e-5
    ) -> dict[str, Any]:
        """
        Compare Python and C++ outputs.

        Args:
            function_name: Name of the function
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Dictionary with comparison results
        """
        python_file = self.results_dir / f"{function_name}_python.npy"
        cpp_file = self.results_dir / f"{function_name}_cpp.npy"

        validation_error = self._validate_result_files(python_file, cpp_file)
        if validation_error:
            return validation_error

        try:
            python_result = np.load(python_file, allow_pickle=True)
            cpp_result = np.load(cpp_file, allow_pickle=True)

            shape_error = self._check_shape_compatibility(python_result, cpp_result)
            if shape_error:
                return shape_error

            return self._compute_comparison_metrics(
                python_result, cpp_result, rtol, atol
            )

        except Exception as e:
            return {"success": False, "error": f"Comparison failed: {str(e)}"}

    def _validate_result_files(self, python_file: Path, cpp_file: Path) -> dict | None:
        """Validate that result files exist."""
        if not python_file.exists():
            return {
                "success": False,
                "error": f"Python result not found: {python_file}",
            }

        if not cpp_file.exists():
            return {"success": False, "error": f"C++ result not found: {cpp_file}"}

        return None

    def _check_shape_compatibility(
        self, python_result: np.ndarray, cpp_result: np.ndarray
    ) -> dict | None:
        """Check if array shapes match."""
        if python_result.shape != cpp_result.shape:
            return {
                "success": False,
                "error": f"Shape mismatch: Python {python_result.shape} vs C++ {cpp_result.shape}",
            }
        return None

    def _compare_structured_arrays(
        self,
        python_result: np.ndarray,
        cpp_result: np.ndarray,
        rtol: float,
        atol: float,
    ) -> dict:
        """Compare structured arrays field by field."""
        all_close = True
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
        total_elements = 0

        for field in python_result.dtype.names:
            py_field = python_result[field].astype(float)
            cpp_field = cpp_result[field].astype(float)

            abs_diff = np.abs(py_field - cpp_field)
            max_abs_diff = max(max_abs_diff, float(np.max(abs_diff)))
            mean_abs_diff += float(np.sum(abs_diff))
            total_elements += abs_diff.size

            if not np.allclose(py_field, cpp_field, rtol=rtol, atol=atol):
                all_close = False

        mean_abs_diff /= total_elements if total_elements > 0 else 1

        return {
            "success": True,
            "is_close": all_close,
            "dtype_match": False,  # Dtypes will differ (int64 vs int32)
            "python_dtype": str(python_result.dtype),
            "cpp_dtype": str(cpp_result.dtype),
            "shape": python_result.shape,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_rel_diff": 0.0,
            "mean_rel_diff": 0.0,
            "match_percentage": 100.0 if all_close else 0.0,
            "total_elements": total_elements,
        }

    def _compute_comparison_metrics(
        self,
        python_result: np.ndarray,
        cpp_result: np.ndarray,
        rtol: float,
        atol: float,
    ) -> dict:
        """Compute numerical comparison metrics."""
        if python_result.dtype.names is not None:
            return self._compare_structured_arrays(
                python_result, cpp_result, rtol, atol
            )

        abs_diff = np.abs(python_result - cpp_result)
        rel_diff = abs_diff / (np.abs(python_result) + 1e-10)

        close_elements = np.isclose(python_result, cpp_result, rtol=rtol, atol=atol)
        match_percentage = 100.0 * np.sum(close_elements) / close_elements.size

        return {
            "success": True,
            "is_close": np.allclose(python_result, cpp_result, rtol=rtol, atol=atol),
            "dtype_match": python_result.dtype == cpp_result.dtype,
            "python_dtype": str(python_result.dtype),
            "cpp_dtype": str(cpp_result.dtype),
            "shape": python_result.shape,
            "max_abs_diff": float(np.max(abs_diff)),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "max_rel_diff": float(np.max(rel_diff)),
            "mean_rel_diff": float(np.mean(rel_diff)),
            "match_percentage": float(match_percentage),
            "total_elements": int(close_elements.size),
        }

    def print_comparison(
        self, comparison: dict[str, Any], python_time: float, cpp_time: float
    ):
        """Print formatted comparison results."""
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)

        if not comparison["success"]:
            print(f"❌ Comparison failed: {comparison['error']}")
            return

        self._print_numerical_accuracy(comparison)
        self._print_performance_stats(python_time, cpp_time)
        print("\n" + "=" * 70)

    def _print_numerical_accuracy(self, comparison: dict[str, Any]) -> None:
        """Print numerical accuracy statistics."""
        print("\n📊 Numerical Accuracy:")
        print(f"  Shape: {comparison['shape']}")
        print(f"  Total elements: {comparison['total_elements']:,}")
        print(f"  Match percentage: {comparison['match_percentage']:.2f}%")

        status = "✅ PASSED (within tolerance)" if comparison["is_close"] else "⚠️  DIFFERENCES DETECTED"
        print(f"  Status: {status}")

        print("\n  Absolute differences:")
        print(f"    Max: {comparison['max_abs_diff']:.6e}")
        print(f"    Mean: {comparison['mean_abs_diff']:.6e}")

        print("\n  Relative differences:")
        print(f"    Max: {comparison['max_rel_diff']:.6e}")
        print(f"    Mean: {comparison['mean_rel_diff']:.6e}")

        if not comparison["dtype_match"]:
            print(
                f"\n  ⚠️  Dtype mismatch: Python {comparison['python_dtype']} "
                f"vs C++ {comparison['cpp_dtype']}"
            )

    def _print_performance_stats(self, python_time: float, cpp_time: float) -> None:
        """Print performance comparison statistics."""
        print("\n⚡ Performance Comparison:")
        print(f"  Python execution time: {python_time * 1000:.3f} ms")
        print(f"  C++ execution time: {cpp_time * 1000:.3f} ms")

        if cpp_time <= 0:
            return

        speedup = python_time / cpp_time
        print(f"  Speedup: {speedup:.2f}x")

        performance_msg = self._get_performance_message(speedup)
        print(f"  {performance_msg}")

    def _get_performance_message(self, speedup: float) -> str:
        """Get performance assessment message based on speedup."""
        if speedup > 1.5:
            return "🚀 C++ is significantly faster!"
        elif speedup > 1.0:
            return "✓ C++ is faster"
        elif speedup > 0.8:
            return "≈ Similar performance"
        else:
            return "⚠️  C++ is slower (possible overhead)"
