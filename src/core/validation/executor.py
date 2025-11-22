"""Code Runners for Python and C++ execution and benchmarking."""

import contextlib
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


class PythonRunner:
    """
    Runs Python preprocessing functions and captures output.
    """

    def __init__(self, results_dir: str = ".build/results"):
        """
        Initialize Python runner.

        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_function(
        self, source_file: str, function_name: str, input_data: Any
    ) -> tuple[Any, float]:
        """
        Run a Python function and measure execution time.

        Args:
            source_file: Path to Python source file
            function_name: Name of function to run
            input_data: Input data for the function

        Returns:
            Tuple of (result, execution_time_seconds)
        """
        func = self._load_and_prepare_function(source_file, function_name)
        self._warmup_function(func, input_data)

        result, execution_time = self._measure_execution(func, input_data)
        self._save_result(result, function_name, "python")

        return result, execution_time

    def _load_and_prepare_function(self, source_file: str, function_name: str) -> Any:
        """Load module and extract function."""
        module = self._load_module_from_file(source_file)
        return self._get_function_from_module(module, function_name, source_file)

    def _load_module_from_file(self, source_file: str) -> Any:
        """Load Python module from file path."""
        spec = importlib.util.spec_from_file_location("module", source_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {source_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["module"] = module
        spec.loader.exec_module(module)
        return module

    def _get_function_from_module(
        self, module: Any, function_name: str, source_file: str
    ) -> Any:
        """Extract function from module."""
        if not hasattr(module, function_name):
            raise ValueError(f"Function '{function_name}' not found in {source_file}")
        return getattr(module, function_name)

    def _warmup_function(self, func: Any, input_data: Any) -> None:
        """Execute warmup run to avoid cold start effects."""
        try:
            _ = func(input_data)
        except Exception as e:
            print(f"Warning: Warmup failed: {e}")

    def _measure_execution(self, func, input_data: Any) -> tuple[Any, float]:
        """Measure function execution time."""
        start_time = time.perf_counter()
        result = func(input_data)
        end_time = time.perf_counter()
        return result, end_time - start_time

    def _save_result(self, result: Any, function_name: str, suffix: str) -> None:
        """Save result to file."""
        output_path = self.results_dir / f"{function_name}_{suffix}.npy"

        if isinstance(result, np.ndarray):
            np.save(output_path, result)
        elif isinstance(result, tuple):
            self._save_tuple_results(result, function_name, suffix)
        else:
            self._try_save_as_numpy(result, output_path)

    def _save_tuple_results(
        self, result: tuple, function_name: str, suffix: str
    ) -> None:
        """Save tuple of scalars as structured array."""
        output_path = self.results_dir / f"{function_name}_{suffix}.npy"

        if len(result) == 2:
            dtype1 = np.dtype(type(result[0])).type
            dtype2 = np.dtype(type(result[1])).type
            dt = np.dtype([("f0", dtype1), ("f1", dtype2)])
            structured = np.array((result[0], result[1]), dtype=dt)
            np.save(output_path, structured)
        else:
            # Fallback for other tuple sizes
            np.save(output_path, np.array(result))

    def _try_save_as_numpy(self, result: Any, output_path: Path) -> None:
        """Attempt to convert and save result as numpy array."""
        with contextlib.suppress(Exception):
            np.save(output_path, np.array(result))


class CppRunner:
    """
    Builds and runs C++ code.
    """

    def __init__(self, results_dir: str = ".build/results"):
        """
        Initialize C++ runner.

        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def build(self, project_dir: str, verbose: bool = False) -> bool:
        """
        Build C++ project using CMake.

        Args:
            project_dir: Path to C++ project directory
            verbose: Print build output

        Returns:
            True if build succeeded
        """
        project_path = Path(project_dir)
        build_dir = project_path / "build"
        build_dir.mkdir(exist_ok=True)

        try:
            if not self._run_cmake(build_dir):
                return False

            if not self._run_make(build_dir):
                return False

            if verbose:
                print("Build successful")

            return True

        except subprocess.TimeoutExpired:
            print("Build timeout")
            return False
        except Exception as e:
            print(f"Build error: {e}")
            return False

    def _run_cmake(self, build_dir: Path) -> bool:
        """Execute CMake configuration."""
        cmake_result = subprocess.run(
            ["cmake", ".."], cwd=build_dir, capture_output=True, text=True, timeout=60
        )

        if cmake_result.returncode != 0:
            print("CMake failed:")
            print(cmake_result.stderr)
            return False

        return True

    def _run_make(self, build_dir: Path) -> bool:
        """Execute make compilation."""
        make_result = subprocess.run(
            ["make", "-j4"], cwd=build_dir, capture_output=True, text=True, timeout=300
        )

        if make_result.returncode != 0:
            print("Make failed:")
            print(make_result.stderr)
            return False

        return True

    def run_executable(
        self, project_dir: str, project_name: str, input_data: str, iterations: int = 10
    ) -> tuple[bool, float]:
        """
        Run C++ executable and measure execution time.

        Args:
            project_dir: Path to C++ project directory
            project_name: Name of the executable
            input_data: Input file path or data
            iterations: Number of iterations for timing

        Returns:
            Tuple of (success, average_execution_time_seconds)
        """
        executable = Path(project_dir) / "build" / project_name
        project_path = Path(project_dir)

        if not executable.exists():
            print(f"Executable not found: {executable}")
            return False, 0.0

        input_path = self._resolve_input_path(input_data)
        return self._execute_and_measure(
            executable, input_path, project_path, iterations
        )

    def _execute_and_measure(
        self, executable: Path, input_path: str, project_path: Path, iterations: int
    ) -> tuple[bool, float]:
        """Execute binary and measure performance."""
        try:
            self._warmup_executable(executable, input_path, project_path)

            avg_time = self._measure_execution_time(
                executable, input_path, project_path, iterations
            )

            self._copy_results_to_global_dir(project_path)
            return True, avg_time

        except subprocess.TimeoutExpired:
            print("Execution timeout")
            return False, 0.0
        except Exception as e:
            print(f"Execution error: {e}")
            return False, 0.0

    def _resolve_input_path(self, input_data: str) -> str:
        """Convert relative input path to absolute path."""
        if os.path.exists(input_data):
            return str(Path(input_data).absolute())
        return input_data

    def _warmup_executable(
        self, executable: Path, input_path: str, project_path: Path
    ) -> None:
        """Execute warmup run of C++ executable."""
        subprocess.run(
            [str(executable.absolute()), input_path],
            cwd=project_path / "build",
            capture_output=True,
            timeout=10,
        )

    def _measure_execution_time(
        self, executable: Path, input_path: str, project_path: Path, iterations: int
    ) -> float:
        """Measure average execution time over multiple iterations."""
        times = []
        for _ in range(iterations):
            iteration_time = self._execute_single_iteration(
                executable, input_path, project_path
            )
            times.append(iteration_time)
        return sum(times) / len(times)

    def _execute_single_iteration(
        self, executable: Path, input_path: str, project_path: Path
    ) -> float:
        """Execute single iteration and return execution time."""
        start_time = time.perf_counter()
        result = subprocess.run(
            [str(executable.absolute()), input_path],
            cwd=project_path / "build",
            capture_output=True,
            text=True,
            timeout=10,
        )
        end_time = time.perf_counter()

        if result.returncode != 0:
            raise RuntimeError(f"Execution failed: {result.stderr}")

        return end_time - start_time

    def _copy_results_to_global_dir(self, project_path: Path) -> None:
        """Copy result files from project directory to global results directory."""
        import shutil

        project_results = project_path / "build" / "results"
        if project_results.exists():
            for file in project_results.glob("*.npy"):
                shutil.copy(file, self.results_dir / file.name)
