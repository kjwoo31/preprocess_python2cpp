#!/usr/bin/env python3
"""Analyze Python files for CLAUDE.md compliance violations."""

import ast
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FunctionMetrics:
    file_path: str
    name: str
    line_start: int
    line_end: int
    body_lines: int

    @property
    def violates_limit(self) -> bool:
        return self.body_lines > 50


def count_function_body_lines(func_node: ast.FunctionDef, source_lines: list[str]) -> int:
    """Count actual body lines excluding signature and docstring."""
    if not func_node.body:
        return 0

    first_stmt = func_node.body[0]
    start_line = first_stmt.lineno

    if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
        if isinstance(first_stmt.value.value, str):
            if len(func_node.body) > 1:
                start_line = func_node.body[1].lineno
            else:
                return 0

    end_line = func_node.end_lineno or func_node.lineno
    return end_line - start_line + 1


def analyze_file(file_path: Path) -> list[FunctionMetrics]:
    """Analyze single Python file for function metrics."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
        source_lines = source.split('\n')

        metrics = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body_lines = count_function_body_lines(node, source_lines)
                metrics.append(FunctionMetrics(
                    file_path=str(file_path),
                    name=node.name,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    body_lines=body_lines
                ))

        return metrics
    except Exception:
        return []


def main():
    src_dir = Path("/mnt/c/Users/kimjw/Downloads/preprocess_python2cpp/src")
    all_metrics = []

    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        metrics = analyze_file(py_file)
        all_metrics.extend(metrics)

    violations = [m for m in all_metrics if m.violates_limit]
    violations.sort(key=lambda x: x.body_lines, reverse=True)

    print(f"Total functions: {len(all_metrics)}")
    print(f"Violations (>50 lines): {len(violations)}\n")

    if violations:
        print("TOP VIOLATIONS:")
        print("-" * 80)
        for m in violations[:20]:
            rel_path = Path(m.file_path).relative_to(src_dir.parent)
            print(f"{m.body_lines:3d} lines | {rel_path}:{m.line_start} | {m.name}")

        print(f"\n... and {len(violations) - 20} more" if len(violations) > 20 else "")


if __name__ == "__main__":
    main()
