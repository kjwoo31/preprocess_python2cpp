"""CLI entry point for Python-to-C++ preprocessing converter."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analysis.parser import PythonASTParser
from core.analysis.inferencer import TypeInferenceEngine
from core.intermediate.schema import IRPipeline, IRInput, IROperation, IROutput, OperationType, TypeHint
from core.generation.generator import CodeGenerator
from core.mapping.database import MappingDatabase
from core.validation.executor import PythonRunner, CppRunner
from core.validation.comparator import ResultComparator


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
    parser.add_argument('--validate', action='store_true', help='Build and validate C++ code')
    parser.add_argument('--test-input', help='Test input file for validation')
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


def build_ir_pipeline(parser: PythonASTParser, func_name: str,
                     tree, type_engine: TypeInferenceEngine) -> IRPipeline:
    """Build IR pipeline from parsed function."""
    func_info = next((f for f in parser.functions if f.name == func_name), None)
    if not func_info:
        raise ValueError(f"Function not found: {func_name}")

    type_context = type_engine.analyze_function(
        next(node for node in tree.body if hasattr(node, 'name') and node.name == func_name)
    )

    inputs = [IRInput(arg, type_context.get(arg, TypeHint('auto')))
              for arg in func_info.args]

    operations = []
    for i, stmt in enumerate(func_info.body):
        ops = _extract_operations_from_statement(stmt, type_context, i)
        operations.extend(ops)

    outputs = _determine_outputs(func_info.body, type_context)
    libraries = list(parser.get_library_usage())

    return IRPipeline(
        name=func_name,
        inputs=inputs,
        operations=operations,
        outputs=outputs,
        metadata={'source_file': parser.__class__.__name__, 'libraries': libraries}
    )


def _extract_operations_from_statement(stmt, type_context: dict,
                                      index: int) -> list[IROperation]:
    """Extract IR operations from statement."""
    import ast

    operations = []

    if isinstance(stmt, ast.Assign):
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            output_var = target.id
            output_type = type_context.get(output_var, TypeHint('auto'))

            if isinstance(stmt.value, ast.Call):
                op = _create_function_call_operation(
                    stmt.value, output_var, output_type, index
                )
                if op:
                    operations.append(op)
            elif isinstance(stmt.value, ast.BinOp):
                op = _create_binary_operation(
                    stmt.value, output_var, output_type, index
                )
                if op:
                    operations.append(op)

    return operations


def _create_binary_operation(binop_node, output_var: str,
                            output_type: TypeHint,
                            index: int) -> Optional[IROperation]:
    """Create binary operation from AST node."""
    import ast
    import re

    def process_operand(node):
        """Process operand, handling special cases like .astype()."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == 'astype':
                    return ast.unparse(node.func.value)
        return ast.unparse(node)

    left = process_operand(binop_node.left)
    right = process_operand(binop_node.right)

    op_map = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.Div: '/',
        ast.Mod: '%'
    }

    operator = op_map.get(type(binop_node.op), '?')

    return IROperation(
        id=f"op_{index + 1}",
        op_type=OperationType.ARITHMETIC,
        output=output_var,
        output_type_hint=output_type,
        operands=[left, right],
        operator=operator
    )


def _create_function_call_operation(call_node, output_var: str,
                                    output_type: TypeHint,
                                    index: int) -> Optional[IROperation]:
    """Create function call operation from AST node."""
    import ast

    func = call_node.func

    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            source_lib = func.value.id
            function_name = func.attr
            source_object = func.value.id
            op_type = OperationType.FUNCTION_CALL
        elif isinstance(func.value, ast.Call):
            # Method call on result of another call (e.g., img.resize(...).astype(...))
            source_object = ast.unparse(func.value)
            source_lib = None
            function_name = func.attr
            op_type = OperationType.METHOD_CALL
        else:
            # Method call on variable (e.g., img.astype(...))
            source_object = ast.unparse(func.value)
            source_lib = None
            function_name = func.attr
            op_type = OperationType.METHOD_CALL
    elif isinstance(func, ast.Name):
        source_lib = None
        source_object = None
        function_name = func.id
        op_type = OperationType.FUNCTION_CALL
    else:
        return None

    args = [ast.unparse(arg) for arg in call_node.args]
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in call_node.keywords}

    op_dict = {
        'id': f"op_{index + 1}",
        'op_type': op_type,
        'output': output_var,
        'output_type_hint': output_type,
        'function': function_name,
        'args': args,
        'kwargs': kwargs
    }

    if source_lib:
        op_dict['source_lib'] = source_lib
    if source_object:
        op_dict['source_object'] = source_object

    return IROperation(**op_dict)


def _determine_outputs(body, type_context: dict) -> list[IROutput]:
    """Determine output variables from function body."""
    import ast

    for stmt in reversed(body):
        if isinstance(stmt, ast.Return) and stmt.value:
            if isinstance(stmt.value, ast.Name):
                var_name = stmt.value.id
                var_type = type_context.get(var_name, TypeHint('auto'))
                return [IROutput(var_name, var_type)]

    return []


def _handle_ir_only_mode(ast_parser: PythonASTParser, functions: list[str],
                        tree) -> int:
    """Handle IR-only mode by generating IR for functions."""
    type_engine = TypeInferenceEngine()
    for func_name in functions:
        pipeline = build_ir_pipeline(ast_parser, func_name, tree, type_engine)
        print(pipeline.to_json())
    return 0


def _process_conversions(functions: list[str], ast_parser: PythonASTParser,
                        tree, generator: CodeGenerator, args) -> list[str]:
    """Process all function conversions and validations."""
    generated_projects = []

    for i, func_name in enumerate(functions, 1):
        if len(functions) > 1:
            print(f"\n{'='*60}")
            print(f"Converting function {i}/{len(functions)}: {func_name}")
            print('='*60)

        project_dir = convert_function(func_name, ast_parser, tree,
                                     generator, args.verbose)
        generated_projects.append(project_dir)

        if args.validate and args.test_input:
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
                    generator: CodeGenerator, verbose: bool) -> str:
    """Convert single function to C++."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Converting: {func_name}")
        print('='*60)

    type_engine = TypeInferenceEngine()
    pipeline = build_ir_pipeline(parser, func_name, tree, type_engine)

    project_dir = generator.generate(pipeline)

    if verbose:
        report = generator.generate_report(pipeline)
        print(report)

    print(f"✓ C++ project generated: {project_dir}")

    return project_dir


def validate_generated_code(project_dir: str, func_name: str, source_file: str,
                           test_input: str, verbose: bool) -> None:
    """Build and validate generated C++ code."""
    print(f"\n{'='*70}")
    print(f"VALIDATING: {func_name}")
    print('='*70)

    cpp_runner = CppRunner()
    python_runner = PythonRunner()
    comparator = ResultComparator()

    print("\n🔨 Building C++ code...")
    if not cpp_runner.build(project_dir, verbose):
        print("❌ Build failed")
        return

    print("✅ Build successful")

    print("\n🐍 Running Python version...")
    try:
        python_result, python_time = python_runner.run_function(
            source_file=source_file,
            function_name=func_name,
            input_data=test_input
        )
        print(f"✅ Python execution: {python_time*1000:.3f} ms")
    except Exception as e:
        print(f"❌ Python execution failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return

    print("\n⚙️  Running C++ version...")
    success, cpp_time = cpp_runner.run_executable(project_dir, func_name, test_input)

    if not success:
        print("❌ C++ execution failed")
        return

    print(f"✅ C++ execution: {cpp_time*1000:.3f} ms")

    print("\n🔍 Comparing results...")
    comparison = comparator.compare_outputs(func_name)
    comparator.print_comparison(comparison, python_time, cpp_time)


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        ast_parser = parse_python_file(args.input, args.verbose)

        with open(args.input) as f:
            import ast
            tree = ast.parse(f.read())

        functions_to_convert = (
            [args.function] if args.function
            else [f.name for f in ast_parser.functions]
        )

        if not functions_to_convert:
            print("No functions found to convert")
            return 1

        if args.verbose:
            print(f"Found {len(functions_to_convert)} function(s) to convert")

        if args.ir_only:
            return _handle_ir_only_mode(ast_parser, functions_to_convert, tree)

        generator = CodeGenerator(
            output_dir=args.output,
            use_llm=args.llm,
            llm_provider=args.llm_provider
        )

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
