"""Execution path tracing to identify actually-called functions."""

import sys
import trace
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class ExecutionTrace:
    """Results from execution tracing."""
    called_functions: set[tuple[str, str]]
    traced_files: set[str]


class ExecutionTracer:
    """
    Trace execution path to identify which functions are actually called.

    Uses sys.settrace to record all function calls during a sample run.
    This allows pruning unused code (e.g., test() functions when running train).
    """

    def __init__(self):
        self.called_functions: set[tuple[str, str]] = set()
        self.traced_files: set[str] = set()
        self.trace_active = False

    def trace_function_execution(
        self,
        target_func: Callable,
        *args,
        **kwargs
    ) -> tuple[Any, ExecutionTrace]:
        """
        Trace execution of a function call.

        Args:
            target_func: Function to trace
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Tuple of (function result, ExecutionTrace)
        """
        self.called_functions.clear()
        self.traced_files.clear()

        original_trace = sys.gettrace()
        sys.settrace(self._trace_calls)
        self.trace_active = True

        try:
            result = target_func(*args, **kwargs)
        finally:
            self.trace_active = False
            sys.settrace(original_trace)

        trace_result = ExecutionTrace(
            called_functions=self.called_functions.copy(),
            traced_files=self.traced_files.copy()
        )

        return result, trace_result

    def trace_module_execution(
        self,
        module_path: Path,
        args: list[str]
    ) -> ExecutionTrace:
        """
        Trace execution of a Python module with command-line arguments.

        Args:
            module_path: Path to Python module to execute
            args: Command-line arguments (e.g., ['--mode', 'train'])

        Returns:
            ExecutionTrace with called functions
        """
        self.called_functions.clear()
        self.traced_files.clear()

        original_argv = sys.argv
        original_trace = sys.gettrace()

        try:
            sys.argv = [str(module_path)] + args
            sys.settrace(self._trace_calls)
            self.trace_active = True

            with open(module_path) as f:
                code = compile(f.read(), str(module_path), 'exec')
                namespace = {'__name__': '__main__', '__file__': str(module_path)}
                exec(code, namespace)

        finally:
            self.trace_active = False
            sys.settrace(original_trace)
            sys.argv = original_argv

        return ExecutionTrace(
            called_functions=self.called_functions.copy(),
            traced_files=self.traced_files.copy()
        )

    def _trace_calls(self, frame, event, arg):
        """Trace callback for sys.settrace."""
        if event != 'call':
            return self._trace_calls

        code = frame.f_code
        filename = code.co_filename
        func_name = code.co_name

        if self._should_trace_file(filename):
            self.called_functions.add((filename, func_name))
            self.traced_files.add(filename)

        return self._trace_calls

    def _should_trace_file(self, filename: str) -> bool:
        """Determine if file should be traced (exclude stdlib)."""
        if '<' in filename:
            return False

        path = Path(filename)

        stdlib_markers = ['site-packages', 'dist-packages', 'lib/python']
        if any(marker in str(path) for marker in stdlib_markers):
            return False

        if '/usr/lib/' in str(path) or '\\lib\\' in str(path):
            return False

        return True

    def filter_by_traced_functions(
        self,
        all_functions: list[tuple[str, str]],
        trace: ExecutionTrace
    ) -> list[tuple[str, str]]:
        """
        Filter function list to only include traced functions.

        Args:
            all_functions: All available functions [(file, func_name), ...]
            trace: Execution trace results

        Returns:
            Filtered list of functions that were actually called
        """
        normalized_trace = set()
        for filepath, func_name in trace.called_functions:
            normalized_path = str(Path(filepath).resolve())
            normalized_trace.add((normalized_path, func_name))

        filtered = []
        for filepath, func_name in all_functions:
            normalized_path = str(Path(filepath).resolve())
            if (normalized_path, func_name) in normalized_trace:
                filtered.append((filepath, func_name))

        return filtered
