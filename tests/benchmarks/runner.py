"""
Benchmark Runner

Runs Python preprocessing functions and saves ground truth outputs.
"""

import argparse
import importlib.util
import pickle
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np


class BenchmarkRunner:
    """
    Runs Python functions and saves their outputs for validation.
    """

    def __init__(self, output_dir: str = "benchmark_outputs"):
        """
        Initialize benchmark runner.

        Args:
            output_dir: Directory to save benchmark outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_function_from_file(self, file_path: str, function_name: str) -> Callable:
        """
        Load a Python function from a file.

        Args:
            file_path: Path to Python file
            function_name: Name of function to load

        Returns:
            Function object
        """
        spec = importlib.util.spec_from_file_location("module", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, function_name):
            raise ValueError(f"Function '{function_name}' not found in {file_path}")

        return getattr(module, function_name)

    def run_and_save(self, func: Callable, inputs: dict, output_name: str):
        """
        Run a function and save its output.

        Args:
            func: Function to run
            inputs: Dictionary of input arguments
            output_name: Name for the output file
        """
        print(f"Running benchmark: {output_name}")

        # Run function
        result = func(**inputs)

        # Save result
        output_path = self.output_dir / f"{output_name}.npy"

        if isinstance(result, np.ndarray):
            np.save(output_path, result)
            print(f"Saved numpy array: shape={result.shape}, dtype={result.dtype}")
        elif isinstance(result, tuple):
            # Save tuple of arrays
            output_path = self.output_dir / f"{output_name}.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Saved tuple of {len(result)} items")
        else:
            # Save as pickle
            output_path = self.output_dir / f"{output_name}.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(result, f)
            print("Saved result as pickle")

        print(f"Output saved to: {output_path}")

        return result

    def save_inputs(self, inputs: dict, input_name: str):
        """
        Save input data for later use.

        Args:
            inputs: Dictionary of inputs
            input_name: Name for the input file
        """
        output_path = self.output_dir / f"{input_name}_inputs.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(inputs, f)
        print(f"Inputs saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Python benchmark and save outputs"
    )
    parser.add_argument(
        "--file", required=True, help="Python file containing the function"
    )
    parser.add_argument("--function", required=True, help="Function name to benchmark")
    parser.add_argument("--output-name", required=True, help="Name for output file")
    parser.add_argument(
        "--output-dir", default="benchmark_outputs", help="Output directory"
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(output_dir=args.output_dir)

    # Load function
    runner.load_function_from_file(args.file, args.function)

    # For now, this is a basic template
    # In practice, you'd need to specify inputs based on the function
    print(
        "Note: You need to modify this script to provide appropriate inputs for your function"
    )


if __name__ == "__main__":
    main()
