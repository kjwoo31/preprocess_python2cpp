"""CLI entry point for Python-to-C++ preprocessing converter."""

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analysis.inferencer import TypeInferenceEngine
from core.analysis.parser import PythonASTParser
from core.generation.generator import CodeGenerator
from core.intermediate.builder import IRBuilder
from core.intermediate.schema import IRPipeline
from core.validation.comparator import ResultComparator
from core.validation.executor import CppRunner, PythonRunner


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert Python preprocessing code to C++",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-i", "--input", required=True, help="Input Python file")
    parser.add_argument(
        "-o", "--output", default=".build/output", help="Output directory"
    )
    parser.add_argument("-f", "--function", help="Specific function to convert")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--ir-only", action="store_true", help="Generate IR only")
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip validation (default: validate)"
    )
    parser.add_argument("--test-input", help="Test input file for validation")
    parser.add_argument(
        "--recursive", action="store_true", help="Multi-file dependency resolution"
    )
    parser.add_argument(
        "--llm", action="store_true", help="Use LLM for unmapped operations"
    )
    parser.add_argument(
        "--llm-provider",
        default="fallback",
        choices=["openai", "anthropic", "fallback"],
        help="LLM provider (default: fallback = OpenAI first, then Anthropic)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max iterations for compilable solution (default: 3)",
    )
    parser.add_argument(
        "--compile-check",
        action="store_true",
        help="Verify LLM-generated code compiles before accepting",
    )

    return parser


def parse_python_file(file_path: str, verbose: bool = False) -> PythonASTParser:
    """Parse Python source file."""
    source_path = Path(file_path)

    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(source_path) as f:
        source_code = f.read()

    if verbose:
        print(f"Reading Python source: {file_path}")

    parser = PythonASTParser()
    tree = parser.parse(source_code)

    return source_code, parser, tree


def _handle_ir_only_mode(
    ast_parser: PythonASTParser, functions: list[str], tree: ast.Module
) -> int:
    """Handle IR-only mode by generating IR for functions."""
    type_engine = TypeInferenceEngine()
    ir_builder = IRBuilder()
    for func_name in functions:
        pipeline = ir_builder.build_pipeline(ast_parser, func_name, tree, type_engine)
        print(pipeline.to_json())
    return 0


def _process_conversions(
    functions: list[str],
    ast_parser: PythonASTParser,
    tree: ast.Module,
    generator: CodeGenerator,
    args: argparse.Namespace,
) -> list[str]:
    """Process all function conversions and validations."""
    type_engine = TypeInferenceEngine()
    ir_builder = IRBuilder()

    main_functions = ast_parser.get_main_block_functions(tree)
    functions_to_convert = _select_functions_to_convert(
        functions, main_functions, args.verbose
    )

    pipelines = _build_all_pipelines(
        functions_to_convert, ast_parser, tree, ir_builder, type_engine
    )

    main_block_operations = _extract_main_block_operations(
        ast_parser, tree, ir_builder, type_engine, args.verbose
    )

    project_dir = _generate_and_report_project(
        generator, pipelines, main_functions, main_block_operations, args
    )

    _validate_if_requested(
        project_dir, functions_to_convert, args, Path(args.input).stem
    )

    return [project_dir]


def _select_functions_to_convert(
    functions: list[str], main_functions: list[str], verbose: bool
) -> list[str]:
    """Select which functions to convert based on __main__ block."""
    if verbose:
        print(f"All functions found: {', '.join(functions)}")
        if main_functions:
            print(f"Functions in __main__ block: {', '.join(main_functions)}")

    if not main_functions:
        return functions

    functions_to_convert = [f for f in functions if f in main_functions]
    if verbose:
        print(f"Converting only: {', '.join(functions_to_convert)}")

    return functions_to_convert


def _build_all_pipelines(
    functions_to_convert: list[str],
    ast_parser: PythonASTParser,
    tree: ast.Module,
    ir_builder: IRBuilder,
    type_engine: TypeInferenceEngine,
) -> list:
    """Build IR pipelines for all functions to convert."""
    pipelines = []
    for func_name in functions_to_convert:
        pipeline = ir_builder.build_pipeline(ast_parser, func_name, tree, type_engine)
        pipelines.append(pipeline)
    return pipelines


def _extract_main_block_operations(
    ast_parser: PythonASTParser,
    tree: ast.Module,
    ir_builder: IRBuilder,
    type_engine: TypeInferenceEngine,
    verbose: bool,
) -> list:
    """Extract operations from __main__ block."""
    main_block_body = ast_parser.get_main_block_body(tree)
    if not main_block_body:
        return []

    main_block_operations = ir_builder.build_main_block_operations(
        main_block_body, type_engine
    )

    if verbose:
        print(f"Main block operations: {len(main_block_operations)}")

    return main_block_operations


def _generate_and_report_project(
    generator: CodeGenerator,
    pipelines: list,
    main_functions: list[str],
    main_block_operations: list,
    args: argparse.Namespace,
) -> str:
    """Generate C++ project and print reports."""
    source_file_name = Path(args.input).stem
    project_dir = generator.generate_multi_function(
        pipelines, source_file_name, main_functions, main_block_operations
    )

    print(f"✓ C++ project generated: {project_dir}")

    if args.verbose:
        for pipeline in pipelines:
            report = generator.generate_report(pipeline)
            print(report)

    return project_dir


def _validate_if_requested(
    project_dir: str, functions_to_convert: list[str], args: argparse.Namespace, source_file_name: str
) -> None:
    """Run validation if requested by user."""
    should_validate = not args.no_validate
    if should_validate and args.test_input:
        validate_multi_functions(
            project_dir,
            functions_to_convert,
            args.input,
            args.test_input,
            args.verbose,
            source_file_name,
        )


def _print_summary(generated_projects: list[str], num_functions: int) -> None:
    """Print conversion summary."""
    print(f"\n{'=' * 60}")
    print(f"Summary: {num_functions} function(s) converted to C++")
    print("=" * 60)
    for proj in generated_projects:
        print(f"  Project: {proj}")


def validate_generated_code(
    project_dir: str,
    func_name: str,
    source_file: str,
    test_input: str,
    verbose: bool,
    executable_name: str | None = None,
) -> None:
    """Build and validate generated C++ code."""
    _print_validation_header(func_name)

    cpp_runner = CppRunner()
    python_runner = PythonRunner()
    comparator = ResultComparator()

    if not _build_cpp_project(cpp_runner, project_dir, verbose):
        return

    python_time = _run_python_version(
        python_runner, source_file, func_name, test_input, verbose
    )
    if python_time is None:
        return

    exec_name = executable_name if executable_name else func_name
    cpp_time = _run_cpp_version(cpp_runner, project_dir, exec_name, test_input)
    if cpp_time is None:
        return

    _compare_results(comparator, func_name, python_time, cpp_time)


def validate_multi_functions(
    project_dir: str,
    func_names: list[str],
    source_file: str,
    test_input: str,
    verbose: bool,
    executable_name: str,
) -> None:
    """Build and validate multiple functions independently."""
    print(f"\n{'=' * 70}")
    print(f"VALIDATING: {len(func_names)} function(s)")
    print("=" * 70)

    cpp_runner = CppRunner()
    python_runner = PythonRunner()
    comparator = ResultComparator()

    if not _build_cpp_project(cpp_runner, project_dir, verbose):
        return

    # Run C++ once (executes all functions)
    print("\n⚙️  Running C++ version (all functions)...")
    success, cpp_time = cpp_runner.run_executable(
        project_dir, executable_name, test_input
    )
    if not success:
        print("❌ C++ execution failed")
        return
    print(f"✅ C++ execution: {cpp_time * 1000:.3f} ms")

    # Validate each function independently
    for i, func_name in enumerate(func_names, 1):
        print(f"\n{'-' * 70}")
        print(f"[{i}/{len(func_names)}] Validating: {func_name}")
        print("-" * 70)

        python_time = _run_python_version(
            python_runner, source_file, func_name, test_input, verbose
        )
        if python_time is None:
            continue

        _compare_results(comparator, func_name, python_time, cpp_time / len(func_names))


def _print_validation_header(func_name: str) -> None:
    """Print validation section header."""
    print(f"\n{'=' * 70}")
    print(f"VALIDATING: {func_name}")
    print("=" * 70)


def _build_cpp_project(cpp_runner: CppRunner, project_dir: str, verbose: bool) -> bool:
    """Build C++ project and return success status."""
    print("\n🔨 Building C++ code...")
    if not cpp_runner.build(project_dir, verbose):
        print("❌ Build failed")
        return False
    print("✅ Build successful")
    return True


def _run_python_version(
    python_runner: PythonRunner,
    source_file: str,
    func_name: str,
    test_input: str,
    verbose: bool,
) -> float | None:
    """Run Python version and return execution time."""
    print("\n🐍 Running Python version...")
    try:
        python_result, python_time = python_runner.run_function(
            source_file=source_file, function_name=func_name, input_data=test_input
        )
        print(f"✅ Python execution: {python_time * 1000:.3f} ms")
        return float(python_time)
    except Exception as e:
        print(f"❌ Python execution failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return None


def _run_cpp_version(
    cpp_runner: CppRunner, project_dir: str, executable_name: str, test_input: str
) -> float | None:
    """Run C++ version and return execution time."""
    print("\n⚙️  Running C++ version...")
    success, cpp_time = cpp_runner.run_executable(
        project_dir, executable_name, test_input
    )

    if not success:
        print("❌ C++ execution failed")
        return None

    print(f"✅ C++ execution: {cpp_time * 1000:.3f} ms")
    return cpp_time


def _compare_results(
    comparator: ResultComparator, func_name: str, python_time: float, cpp_time: float
) -> None:
    """Compare Python and C++ results."""
    print("\n🔍 Comparing results...")
    comparison = comparator.compare_outputs(func_name)
    comparator.print_comparison(comparison, python_time, cpp_time)


def _determine_functions_to_convert(
    args: argparse.Namespace, ast_parser: PythonASTParser
) -> list[str]:
    """Determine which functions to convert based on arguments."""
    functions = (
        [args.function] if args.function else [f.name for f in ast_parser.functions]
    )

    if not functions:
        raise ValueError("No functions found to convert")

    if args.verbose:
        print(f"Found {len(functions)} function(s) to convert")

    return functions


def _create_generator(args: argparse.Namespace) -> CodeGenerator:
    """Create code generator from arguments."""
    return CodeGenerator(
        output_dir=args.output,
        use_llm=args.llm,
        llm_provider=args.llm_provider,
        max_iterations=args.max_iterations,
        compile_check=args.compile_check,
    )


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        source_code, ast_parser, tree = parse_python_file(args.input, args.verbose)

        functions_to_convert = _determine_functions_to_convert(args, ast_parser)

        if args.ir_only:
            return _handle_ir_only_mode(ast_parser, functions_to_convert, tree)

        generator = _create_generator(args)

        main_functions = ast_parser.get_main_block_functions(tree)
        functions_to_convert_count = (
            len([f for f in functions_to_convert if f in main_functions])
            if main_functions
            else len(functions_to_convert)
        )

        generated_projects = _process_conversions(
            functions_to_convert, ast_parser, tree, generator, args
        )

        _print_summary(generated_projects, functions_to_convert_count)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
