"""
Atomic Function Performance Test Suite

Tests individual OpenCV and NumPy functions for granular performance analysis.
"""

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class TestConfig:
    """Configuration for a single function test."""

    name: str
    source_file: str
    function_name: str
    test_input: str
    category: str


@dataclass
class PerformanceResult:
    """Performance test result."""

    function_name: str
    category: str
    python_time_ms: float
    cpp_time_ms: float
    speedup: float
    accuracy_pct: float
    status: str
    build_success: bool


class AtomicPerformanceTracker:
    """Track atomic function performance results."""

    def __init__(self, output_file: str = "ATOMIC_PERFORMANCE.md"):
        self.output_file = Path(output_file)
        self.results: list[PerformanceResult] = []

    def add_result(self, result: PerformanceResult) -> None:
        """Add new performance result."""
        self.results.append(result)

    def save_to_markdown(self) -> None:
        """Save all results to markdown file."""
        if not self.results:
            print("No results to save")
            return

        by_category = {}
        for r in self.results:
            if r.category not in by_category:
                by_category[r.category] = []
            by_category[r.category].append(r)

        content = f"""# Atomic Function Performance Test Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Functions Tested:** {len(self.results)}
- **Successful Builds:** {sum(1 for r in self.results if r.build_success)}
- **Build Failures:** {sum(1 for r in self.results if not r.build_success)}
- **Average Accuracy:** {sum(r.accuracy_pct for r in self.results if r.build_success) / max(1, sum(1 for r in self.results if r.build_success)):.2f}%

## Categories

"""

        for category, results in sorted(by_category.items()):
            successful = [r for r in results if r.build_success]
            failed = [r for r in results if not r.build_success]

            content += f"### {category}\n\n"
            content += f"- **Total:** {len(results)}\n"
            content += f"- **Success:** {len(successful)}\n"
            content += f"- **Failed:** {len(failed)}\n"
            content += f"- **Avg Accuracy:** {sum(r.accuracy_pct for r in successful) / max(1, len(successful)):.2f}%\n\n"

            if successful:
                content += "| Function | Python (ms) | C++ (ms) | Speedup | Accuracy | Status |\n"
                content += "|----------|-------------|----------|---------|----------|--------|\n"

                for result in successful:
                    status_emoji = "✅" if result.status == "PASS" else "⚠️"
                    content += (
                        f"| `{result.function_name}` | "
                        f"{result.python_time_ms:.3f} ms | "
                        f"{result.cpp_time_ms:.3f} ms | "
                        f"{result.speedup:.2f}x | "
                        f"{result.accuracy_pct:.2f}% | "
                        f"{status_emoji} {result.status} |\n"
                    )

                content += "\n"

            if failed:
                content += "**Build Failures:**\n\n"
                for result in failed:
                    content += f"- `{result.function_name}`: {result.status}\n"
                content += "\n"

        content += """## Analysis

### Best Performing Functions

"""
        top_performers = sorted(
            [r for r in self.results if r.build_success and r.speedup > 0],
            key=lambda x: x.accuracy_pct,
            reverse=True,
        )[:10]

        if top_performers:
            content += "| Function | Accuracy | Category |\n"
            content += "|----------|----------|----------|\n"
            for r in top_performers:
                content += f"| `{r.function_name}` | {r.accuracy_pct:.2f}% | {r.category} |\n"
            content += "\n"

        content += """### Failed Functions

"""
        failures = [r for r in self.results if not r.build_success]
        if failures:
            content += "Functions that failed to build or execute:\n\n"
            for r in failures:
                content += f"- `{r.function_name}` ({r.category}): {r.status}\n"
        else:
            content += "No failures! All functions built successfully.\n"

        content += """
## Test Environment

- **Platform:** Linux (WSL2)
- **Compiler:** g++ (C++17)
- **Python:** 3.10+
- **OpenCV:** 4.x
- **Test Image:** config/data/test_image.jpg

## Notes

- Performance measured as average over 10 iterations
- Accuracy measured as percentage of matching array elements
- Speedup = Python time / C++ time
- Status: PASS if accuracy ≥ 75% and build succeeds
- These are individual atomic functions, not composite operations
"""

        self.output_file.write_text(content)
        print(f"\n✅ Atomic performance results saved to {self.output_file}")


def run_test(config: TestConfig) -> PerformanceResult | None:
    """Run performance test for a single function."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {config.function_name} ({config.category})")
    print('=' * 70)

    cmd = [
        "python",
        "src/cli/main.py",
        "-i",
        config.source_file,
        "-f",
        config.function_name,
        "-o",
        ".build/atomic",
        "--test-input",
        config.test_input,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        output = result.stdout
        print(output)

        if result.returncode != 0:
            print(f"❌ Test failed: {result.stderr}")
            return PerformanceResult(
                function_name=config.function_name,
                category=config.category,
                python_time_ms=0.0,
                cpp_time_ms=0.0,
                speedup=0.0,
                accuracy_pct=0.0,
                status="BUILD_FAILED",
                build_success=False,
            )

        return _parse_output(output, config.function_name, config.category)

    except subprocess.TimeoutExpired:
        print(f"❌ Test timeout for {config.function_name}")
        return PerformanceResult(
            function_name=config.function_name,
            category=config.category,
            python_time_ms=0.0,
            cpp_time_ms=0.0,
            speedup=0.0,
            accuracy_pct=0.0,
            status="TIMEOUT",
            build_success=False,
        )
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return None


def _parse_output(output: str, function_name: str, category: str) -> PerformanceResult:
    """Parse CLI output to extract performance metrics."""
    python_time = 0.0
    cpp_time = 0.0
    accuracy = 0.0
    build_success = "Build successful" in output or "✅ Build successful" in output

    for line in output.split("\n"):
        if "Python execution:" in line:
            try:
                python_time = float(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "C++ execution:" in line:
            try:
                cpp_time = float(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "Match percentage:" in line:
            try:
                accuracy = float(line.split(":")[1].strip().replace("%", ""))
            except (ValueError, IndexError):
                pass

    if cpp_time > 0:
        speedup = python_time / cpp_time
    else:
        speedup = 0.0

    if accuracy >= 75.0 and build_success:
        status = "PASS"
    elif build_success:
        status = "LOW_ACCURACY"
    else:
        status = "BUILD_FAILED"

    return PerformanceResult(
        function_name=function_name,
        category=category,
        python_time_ms=python_time,
        cpp_time_ms=cpp_time,
        speedup=speedup,
        accuracy_pct=accuracy,
        status=status,
        build_success=build_success,
    )


def main():
    """Run all atomic function tests."""
    test_configs = [
        # OpenCV Functions
        TestConfig("OpenCV imread", "examples/atomic/opencv_funcs.py", "test_imread", "config/data/test_image.jpg", "OpenCV"),
        TestConfig("OpenCV imread grayscale", "examples/atomic/opencv_funcs.py", "test_imread_grayscale", "config/data/test_image.jpg", "OpenCV"),
        TestConfig("OpenCV resize", "examples/atomic/opencv_funcs.py", "test_resize", "config/data/test_image.jpg", "OpenCV"),
        TestConfig("OpenCV cvtColor RGB", "examples/atomic/opencv_funcs.py", "test_cvtColor", "config/data/test_image.jpg", "OpenCV"),
        TestConfig("OpenCV cvtColor GRAY", "examples/atomic/opencv_funcs.py", "test_cvtColor_gray", "config/data/test_image.jpg", "OpenCV"),
        TestConfig("OpenCV GaussianBlur", "examples/atomic/opencv_funcs.py", "test_GaussianBlur", "config/data/test_image.jpg", "OpenCV"),
        TestConfig("OpenCV threshold", "examples/atomic/opencv_funcs.py", "test_threshold", "config/data/test_image.jpg", "OpenCV"),
        TestConfig("OpenCV Canny", "examples/atomic/opencv_funcs.py", "test_Canny", "config/data/test_image.jpg", "OpenCV"),

        # NumPy Functions
        TestConfig("NumPy astype float32", "examples/atomic/numpy_funcs.py", "test_astype_float32", "config/data/test_image.jpg", "NumPy"),
        TestConfig("NumPy astype uint8", "examples/atomic/numpy_funcs.py", "test_astype_uint8", "config/data/test_image.jpg", "NumPy"),
        TestConfig("NumPy divide", "examples/atomic/numpy_funcs.py", "test_divide_scalar", "config/data/test_image.jpg", "NumPy"),
        TestConfig("NumPy multiply", "examples/atomic/numpy_funcs.py", "test_multiply_scalar", "config/data/test_image.jpg", "NumPy"),
        TestConfig("NumPy reshape", "examples/atomic/numpy_funcs.py", "test_reshape", "config/data/test_image.jpg", "NumPy"),
        TestConfig("NumPy transpose", "examples/atomic/numpy_funcs.py", "test_transpose", "config/data/test_image.jpg", "NumPy"),
        TestConfig("NumPy mean", "examples/atomic/numpy_funcs.py", "test_mean", "config/data/test_image.jpg", "NumPy"),
        TestConfig("NumPy std", "examples/atomic/numpy_funcs.py", "test_std", "config/data/test_image.jpg", "NumPy"),
        TestConfig("NumPy clip", "examples/atomic/numpy_funcs.py", "test_clip", "config/data/test_image.jpg", "NumPy"),
        TestConfig("NumPy subtract", "examples/atomic/numpy_funcs.py", "test_subtract", "config/data/test_image.jpg", "NumPy"),
        TestConfig("NumPy add", "examples/atomic/numpy_funcs.py", "test_add", "config/data/test_image.jpg", "NumPy"),

        # Type Conversions
        TestConfig("Type int", "examples/atomic/type_funcs.py", "test_int_conversion", "config/data/test_image.jpg", "Type Conversion"),
        TestConfig("Type float", "examples/atomic/type_funcs.py", "test_float_conversion", "config/data/test_image.jpg", "Type Conversion"),
        TestConfig("Type bool", "examples/atomic/type_funcs.py", "test_bool_operation", "config/data/test_image.jpg", "Type Conversion"),
        TestConfig("Type tuple", "examples/atomic/type_funcs.py", "test_tuple_return", "config/data/test_image.jpg", "Type Conversion"),
        TestConfig("Type max", "examples/atomic/type_funcs.py", "test_max_value", "config/data/test_image.jpg", "Type Conversion"),
        TestConfig("Type min", "examples/atomic/type_funcs.py", "test_min_value", "config/data/test_image.jpg", "Type Conversion"),
    ]

    tracker = AtomicPerformanceTracker()

    print("=" * 70)
    print("Atomic Function Performance Test Suite")
    print("=" * 70)
    print(f"Total functions to test: {len(test_configs)}")

    for config in test_configs:
        result = run_test(config)
        if result:
            tracker.add_result(result)

    tracker.save_to_markdown()

    print("\n" + "=" * 70)
    print("Atomic Performance Testing Complete")
    print("=" * 70)
    print(f"Results saved to: ATOMIC_PERFORMANCE.md")


if __name__ == "__main__":
    main()
