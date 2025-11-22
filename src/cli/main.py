"""CLI entry point for Python-to-C++ preprocessing converter."""

import argparse
import ast
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analysis.inferencer import TypeInferenceEngine
from core.analysis.parser import PythonASTParser
from core.analysis.separator import PipelineSeparator
from core.generation.generator import CodeGenerator
from core.intermediate.builder import IRBuilder
from core.intermediate.schema import IRPipeline
from core.validation.comparator import ResultComparator
from core.validation.executor import CppRunner, PythonRunner


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Convert Python preprocessing code to C++',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-i', '--input', required=True, help='Input Python file')
    parser.add_argument('-o', '--output', default='.build/output', help='Output directory')
    parser.add_argument('-f', '--function', help='Specific function to convert')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--ir-only', action='store_true', help='Generate IR only')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation (default: validate)')
    parser.add_argument('--test-input', required=True, help='Test input file for validation')
    parser.add_argument('--pipeline', action='store_true', help='Split conversion mode (Pre/Inf/Post)')
    parser.add_argument('--recursive', action='store_true', help='Multi-file dependency resolution')
    parser.add_argument('--llm', action='store_true', help='Use LLM for unmapped operations')
    parser.add_argument('--llm-provider', default='vertex',
                       choices=['openai', 'anthropic', 'vertex'],
                       help='LLM provider (default: vertex = Anthropic via Google Cloud Vertex AI)')

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
    parser.parse(source_code)

    return parser


def _handle_ir_only_mode(ast_parser: PythonASTParser, functions: list[str],
                        tree, ir_builder: IRBuilder) -> int:
    """Handle IR-only mode by generating IR for functions."""
    type_engine = TypeInferenceEngine()
    for func_name in functions:
        pipeline = ir_builder.build_pipeline(ast_parser, func_name, tree, type_engine)
        print(pipeline.to_json())
    return 0


def _process_conversions(functions: list[str], ast_parser: PythonASTParser,
                        tree, generator: CodeGenerator, args) -> list[str]:
    """Process all function conversions and validations."""
    generated_projects = []
    ir_builder = IRBuilder()

    for i, func_name in enumerate(functions, 1):
        if len(functions) > 1:
            print(f"\n{'='*60}")
            print(f"Converting function {i}/{len(functions)}: {func_name}")
            print('='*60)

        project_dir = convert_function(func_name, ast_parser, tree,
                                     generator, args.verbose, ir_builder)
        generated_projects.append(project_dir)

        should_validate = not args.no_validate
        if should_validate and args.test_input:
            validate_generated_code(project_dir, func_name, args.input,
                                   args.test_input, args.verbose)

    return generated_projects


def _print_summary(generated_projects: list[str]):
    """Print conversion summary."""
    print(f"\n{'='*60}")
    print(f"Summary: {len(generated_projects)} function(s) converted successfully")
    print('='*60)
    for proj in generated_projects:
        print(f"  - {proj}")


def convert_function(func_name: str, parser: PythonASTParser, tree,
                    generator: CodeGenerator, verbose: bool,
                    ir_builder: IRBuilder) -> str:
    """Convert single function to C++."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Converting: {func_name}")
        print('='*60)

    type_engine = TypeInferenceEngine()
    pipeline = ir_builder.build_pipeline(parser, func_name, tree, type_engine)

    project_dir = generator.generate(pipeline)

    if verbose:
        report = generator.generate_report(pipeline)
        print(report)

    print(f"✓ C++ project generated: {project_dir}")

    return project_dir


def process_pipeline_conversion(source_file: str, generator: CodeGenerator,
                                verbose: bool, test_input: str | None = None,
                                should_validate: bool = True) -> str:
    """Process pipeline conversion (Pre/Inf/Post split)."""
    source_code, tree = _load_source_code(source_file)
    separated = PipelineSeparator().separate(source_code, tree)

    pre_pipeline, inf_pipeline, post_pipeline = _build_pipelines_from_separation(
        separated, source_code
    )

    func_name = _extract_function_name(tree) or "pipeline"

    project_dir = generator.generate_pipeline(
        pre_pipeline, inf_pipeline, post_pipeline,
        project_name=func_name, function_name=func_name
    )

    print(f"✓ Pipeline project generated: {project_dir}")

    if should_validate and test_input:
        validate_generated_code(project_dir, func_name, source_file,
                               test_input, verbose, project_name=func_name)

    return project_dir


def _load_source_code(source_file: str) -> tuple:
    """Load and parse Python source file."""
    with open(source_file) as f:
        source_code = f.read()
        tree = __import__('ast').parse(source_code)
    return source_code, tree


def _build_pipelines_from_separation(separated, source_code: str) -> tuple:
    """Build IR pipelines from separated segments."""
    type_engine = TypeInferenceEngine()
    ast_parser = PythonASTParser()
    ast_parser.parse(source_code)
    ir_builder = IRBuilder()

    ir_builder = IRBuilder()
    pre_pipeline = _build_preprocess_pipeline(separated, type_engine, ast_parser, ir_builder)
    inf_pipeline = _build_inference_pipeline(separated, type_engine, ast_parser, ir_builder)
    post_pipeline = _build_postprocess_pipeline(separated, type_engine, ast_parser, ir_builder)

    return pre_pipeline, inf_pipeline, post_pipeline


def _build_preprocess_pipeline(separated, type_engine, ast_parser, ir_builder: IRBuilder) -> Optional[IRPipeline]:
    """Build preprocessing pipeline if available."""
    if not separated.preprocess:
        return None
    return ir_builder.build_pipeline_from_segment(
        separated.preprocess, "preprocess", type_engine, ast_parser
    )


def _build_inference_pipeline(separated, type_engine, ast_parser, ir_builder: IRBuilder) -> Optional[IRPipeline]:
    """Build inference pipeline if available."""
    if not separated.inference:
        return None

    inf_pipeline = ir_builder.build_pipeline_from_segment(
        separated.inference, "inference", type_engine, ast_parser
    )
    inference_code = '\n'.join(ast.unparse(stmt) for stmt in separated.inference.body)
    inf_pipeline.metadata['inference_code'] = inference_code
    return inf_pipeline


def _build_postprocess_pipeline(separated, type_engine, ast_parser, ir_builder: IRBuilder) -> Optional[IRPipeline]:
    """Build postprocessing pipeline if available."""
    if not separated.postprocess:
        return None
    return ir_builder.build_pipeline_from_segment(
        separated.postprocess, "postprocess", type_engine, ast_parser
    )


def _extract_function_name(tree) -> str | None:
    """Extract function name from AST."""
    for node in tree.body:
        if isinstance(node, __import__('ast').FunctionDef):
            return node.name
    return None


def validate_generated_code(project_dir: str, func_name: str, source_file: str,
                           test_input: str, verbose: bool, project_name: str | None = None) -> None:
    """Build and validate generated C++ code."""
    _print_validation_header(func_name)

    cpp_runner = CppRunner()
    python_runner = PythonRunner()
    comparator = ResultComparator()

    if not _build_cpp_project(cpp_runner, project_dir, verbose):
        return

    executable_name = project_name if project_name else func_name

    python_time = _run_python_version(python_runner, source_file, func_name, test_input, verbose)
    if python_time is None:
        return

    cpp_time = _run_cpp_version(cpp_runner, project_dir, executable_name, test_input)
    if cpp_time is None:
        return

    _compare_results(comparator, func_name, python_time, cpp_time)


def _print_validation_header(func_name: str):
    """Print validation section header."""
    print(f"\n{'='*70}")
    print(f"VALIDATING: {func_name}")
    print('='*70)


def _build_cpp_project(cpp_runner: CppRunner, project_dir: str, verbose: bool) -> bool:
    """Build C++ project and return success status."""
    print("\n🔨 Building C++ code...")
    if not cpp_runner.build(project_dir, verbose):
        print("❌ Build failed")
        return False
    print("✅ Build successful")
    return True


def _run_python_version(python_runner: PythonRunner, source_file: str,
                       func_name: str, test_input: str, verbose: bool) -> float | None:
    """Run Python version and return execution time."""
    print("\n🐍 Running Python version...")
    try:
        python_result, python_time = python_runner.run_function(
            source_file=source_file,
            function_name=func_name,
            input_data=test_input
        )
        print(f"✅ Python execution: {python_time*1000:.3f} ms")
        return python_time
    except Exception as e:
        print(f"❌ Python execution failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def _run_cpp_version(cpp_runner: CppRunner, project_dir: str,
                    executable_name: str, test_input: str) -> float | None:
    """Run C++ version and return execution time."""
    print("\n⚙️  Running C++ version...")
    success, cpp_time = cpp_runner.run_executable(project_dir, executable_name, test_input)

    if not success:
        print("❌ C++ execution failed")
        return None

    print(f"✅ C++ execution: {cpp_time*1000:.3f} ms")
    return cpp_time


def _compare_results(comparator: ResultComparator, func_name: str,
                    python_time: float, cpp_time: float):
    """Compare Python and C++ results."""
    print("\n🔍 Comparing results...")
    comparison = comparator.compare_outputs(func_name)
    comparator.print_comparison(comparison, python_time, cpp_time)


def _determine_functions_to_convert(args, ast_parser) -> list[str]:
    """Determine which functions to convert based on arguments."""
    functions = (
        [args.function] if args.function
        else [f.name for f in ast_parser.functions]
    )

    if not functions:
        raise ValueError("No functions found to convert")

    if args.verbose:
        print(f"Found {len(functions)} function(s) to convert")

    return functions


def _create_generator(args) -> CodeGenerator:
    """Create code generator from arguments."""
    return CodeGenerator(
        output_dir=args.output,
        use_llm=args.llm,
        llm_provider=args.llm_provider
    )


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        ast_parser = parse_python_file(args.input, args.verbose)

        with open(args.input) as f:
            import ast
            tree = ast.parse(f.read())

        functions_to_convert = _determine_functions_to_convert(args, ast_parser)

        if args.ir_only:
            ir_builder = IRBuilder()
            return _handle_ir_only_mode(ast_parser, functions_to_convert, tree, ir_builder)

        generator = _create_generator(args)

        if args.pipeline:
            should_validate = not args.no_validate
            project_dir = process_pipeline_conversion(
                args.input, generator, args.verbose,
                args.test_input, should_validate
            )
            print(f"\n{'='*60}")
            print(f"Pipeline project generated: {project_dir}")
            print('='*60)
            return 0

        generated_projects = _process_conversions(
            functions_to_convert, ast_parser, tree, generator, args
        )

        _print_summary(generated_projects)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
