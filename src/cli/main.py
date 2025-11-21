"""CLI entry point for Python-to-C++ preprocessing converter."""

import argparse
import ast
import sys
from pathlib import Path
from typing import Optional

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analysis.parser import PythonASTParser
from core.analysis.inferencer import TypeInferenceEngine
from core.analysis.separator import PipelineSeparator
from core.intermediate.schema import IRPipeline, IRInput, IROperation, IROutput, OperationType, TypeHint
from core.generation.generator import CodeGenerator
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


def build_ir_pipeline(parser: PythonASTParser, func_name: str,
                     tree: ast.Module, type_engine: TypeInferenceEngine) -> IRPipeline:
    """Build IR pipeline from parsed function."""
    func_info = _find_function_info(parser, func_name)
    type_context = _build_type_context(tree, func_name, type_engine)

    inputs = _build_pipeline_inputs(func_info, type_context)
    operations = _build_pipeline_operations(func_info, type_context)
    outputs = _determine_outputs(func_info.body, type_context)
    libraries = list(parser.get_library_usage())

    return IRPipeline(
        name=func_name,
        inputs=inputs,
        operations=operations,
        outputs=outputs,
        metadata={'source_file': parser.__class__.__name__, 'libraries': libraries}
    )


def _find_function_info(parser: PythonASTParser, func_name: str):
    """Find function information from parser."""
    func_info = next((f for f in parser.functions if f.name == func_name), None)
    if not func_info:
        raise ValueError(f"Function not found: {func_name}")
    return func_info


def _build_type_context(tree: ast.Module, func_name: str,
                       type_engine: TypeInferenceEngine) -> dict[str, TypeHint]:
    """Build type context for function."""
    func_node = next(
        (node for node in tree.body
         if hasattr(node, 'name') and node.name == func_name),
        None
    )
    if not func_node:
        return {}
    return type_engine.analyze_function(func_node)


def _build_pipeline_inputs(func_info, type_context: dict) -> list[IRInput]:
    """Build IR inputs from function arguments."""
    return [IRInput(arg, type_context.get(arg, TypeHint('auto')))
            for arg in func_info.args]


def _build_pipeline_operations(func_info, type_context: dict) -> list[IROperation]:
    """Build IR operations from function body."""
    operations = []
    for i, stmt in enumerate(func_info.body):
        ops = _extract_operations_from_statement(stmt, type_context, i)
        operations.extend(ops)
    return operations


def _extract_operations_from_statement(stmt, type_context: dict,
                                      index: int) -> list[IROperation]:
    """Extract IR operations from statement."""
    operations = []

    if isinstance(stmt, ast.Assign):
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            output_var = target.id
            output_type = type_context.get(output_var, TypeHint('auto'))
            op = _create_assignment_operation(stmt.value, output_var, output_type, index)
            if op:
                operations.append(op)
    elif isinstance(stmt, ast.If):
        op = _create_conditional_operation(stmt, type_context, index)
        if op:
            operations.append(op)
    elif isinstance(stmt, (ast.For, ast.While)):
        op = _create_loop_operation(stmt, type_context, index)
        if op:
            operations.append(op)

    return operations


def _create_assignment_operation(value_node, output_var: str,
                                 output_type: TypeHint, index: int) -> Optional[IROperation]:
    """Create operation from assignment value node."""
    if isinstance(value_node, ast.Call):
        return _create_function_call_operation(value_node, output_var, output_type, index)
    elif isinstance(value_node, ast.BinOp):
        return _create_binary_operation(value_node, output_var, output_type, index)
    elif isinstance(value_node, ast.List):
        return _create_list_assignment(value_node, output_var, output_type, index)
    elif isinstance(value_node, ast.Subscript):
        return _create_subscript_assignment(value_node, output_var, output_type, index)
    elif isinstance(value_node, ast.Name):
        return _create_variable_assignment(value_node, output_var, output_type, index)
    elif isinstance(value_node, ast.Constant):
        return _create_constant_assignment(value_node, output_var, output_type, index)
    return None


def _create_list_assignment(list_node: ast.List, output_var: str,
                           output_type: TypeHint, index: int) -> IROperation:
    """Create C++ vector assignment from Python list."""
    elements = _convert_list_elements(list_node)
    cpp_vector = "std::vector<std::string>{" + ", ".join(elements) + "}"
    return IROperation(
        id=f"op_{index + 1}",
        op_type=OperationType.ASSIGNMENT,
        output=output_var,
        output_type_hint=output_type,
        args=[cpp_vector]
    )


def _convert_list_elements(list_node: ast.List) -> list[str]:
    """Convert Python list elements to C++ representation."""
    elements = []
    for elt in list_node.elts:
        if isinstance(elt, ast.Constant):
            if isinstance(elt.value, str):
                elements.append(f'"{elt.value}"')
            else:
                elements.append(str(elt.value))
        else:
            elements.append(ast.unparse(elt))
    return elements


def _create_subscript_assignment(subscript_node: ast.Subscript, output_var: str,
                                output_type: TypeHint, index: int) -> IROperation:
    """Create array subscript assignment operation."""
    return _create_simple_assignment(output_var, output_type, index, ast.unparse(subscript_node))


def _create_variable_assignment(name_node: ast.Name, output_var: str,
                               output_type: TypeHint, index: int) -> IROperation:
    """Create simple variable assignment operation."""
    return _create_simple_assignment(output_var, output_type, index, name_node.id)


def _create_constant_assignment(const_node: ast.Constant, output_var: str,
                                output_type: TypeHint, index: int) -> IROperation:
    """Create constant value assignment operation."""
    cpp_value = _convert_constant_to_cpp(const_node.value)
    return _create_simple_assignment(output_var, output_type, index, cpp_value)


def _create_simple_assignment(output_var: str, output_type: TypeHint,
                             index: int, value: str) -> IROperation:
    """Create simple assignment IR operation."""
    return IROperation(
        id=f"op_{index + 1}",
        op_type=OperationType.ASSIGNMENT,
        output=output_var,
        output_type_hint=output_type,
        args=[value]
    )


def _create_conditional_operation(if_stmt: ast.If, type_context: dict,
                                   index: int) -> IROperation:
    """Create conditional (if/else) operation."""
    condition = ast.unparse(if_stmt.test)

    true_ops = []
    for i, stmt in enumerate(if_stmt.body):
        ops = _extract_operations_from_statement(stmt, type_context, index * 1000 + i)
        true_ops.extend(ops)

    false_ops = []
    for i, stmt in enumerate(if_stmt.orelse):
        ops = _extract_operations_from_statement(stmt, type_context, index * 1000 + 100 + i)
        false_ops.extend(ops)

    return IROperation(
        id=f"op_{index + 1}",
        op_type=OperationType.CONDITIONAL,
        output="",
        output_type_hint=TypeHint('void'),
        condition=condition,
        true_branch=true_ops,
        false_branch=false_ops
    )


def _create_loop_operation(loop_stmt, type_context: dict, index: int) -> IROperation:
    """Create loop (for/while) operation."""
    if isinstance(loop_stmt, ast.For):
        loop_var = loop_stmt.target.id if isinstance(loop_stmt.target, ast.Name) else ""
        iterable = ast.unparse(loop_stmt.iter)
    else:
        loop_var = ""
        iterable = ast.unparse(loop_stmt.test)

    loop_ops = []
    for i, stmt in enumerate(loop_stmt.body):
        ops = _extract_operations_from_statement(stmt, type_context, index * 1000 + i)
        loop_ops.extend(ops)

    return IROperation(
        id=f"op_{index + 1}",
        op_type=OperationType.LOOP,
        output="",
        output_type_hint=TypeHint('void'),
        loop_var=loop_var,
        iterable=iterable,
        loop_body=loop_ops
    )


def _convert_constant_to_cpp(value) -> str:
    """Convert Python constant to C++ representation."""
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, bool):
        return "true" if value else "false"
    else:
        return str(value)


def _create_binary_operation(binop_node, output_var: str,
                            output_type: TypeHint,
                            index: int) -> Optional[IROperation]:
    """Create binary operation from AST node."""
    left = _process_operand(binop_node.left)
    right = _process_operand(binop_node.right)
    operator = _get_operator_symbol(binop_node.op)

    return IROperation(
        id=f"op_{index + 1}",
        op_type=OperationType.ARITHMETIC,
        output=output_var,
        output_type_hint=output_type,
        operands=[left, right],
        operator=operator
    )


def _process_operand(node) -> str:
    """Process operand, unwrapping astype() calls."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'astype':
                return ast.unparse(node.func.value)
    return ast.unparse(node)


def _get_operator_symbol(op_node) -> str:
    """Convert AST operator to symbol."""
    op_map = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.Div: '/',
        ast.Mod: '%'
    }
    return op_map.get(type(op_node), '?')


def _analyze_function_call_type(func):
    """Analyze function call type and extract metadata."""
    KNOWN_LIBRARIES = {'cv2', 'np', 'numpy', 'torch', 'tf', 'PIL', 'Image'}

    if isinstance(func, ast.Attribute):
        return _analyze_attribute_call(func, KNOWN_LIBRARIES)
    elif isinstance(func, ast.Name):
        return None, func.id, None, OperationType.FUNCTION_CALL
    else:
        return None, None, None, None


def _analyze_attribute_call(func: ast.Attribute, known_libraries: set):
    """Analyze attribute function call."""
    if isinstance(func.value, ast.Name):
        lib_or_var = func.value.id
        if lib_or_var in known_libraries:
            return lib_or_var, func.attr, lib_or_var, OperationType.FUNCTION_CALL
        return None, func.attr, lib_or_var, OperationType.METHOD_CALL

    source_object = ast.unparse(func.value)
    return None, func.attr, source_object, OperationType.METHOD_CALL


def _expand_tuple_args(arg_node):
    """Expand tuple arguments into individual elements."""
    if isinstance(arg_node, ast.Tuple):
        return [_convert_constant(elt) for elt in arg_node.elts]
    return [_convert_constant(arg_node)]


def _convert_constant(node):
    """Convert Python constant to C++ value if it's a known constant."""
    if isinstance(node, ast.Attribute):
        mapped_value = _try_map_attribute_constant(node)
        if mapped_value:
            return mapped_value

    return ast.unparse(node)


def _try_map_attribute_constant(node: ast.Attribute) -> Optional[str]:
    """Try to map attribute constant to C++ equivalent."""
    from core.mapping.database import MappingDatabase

    if not isinstance(node.value, ast.Name):
        return None

    lib = node.value.id
    const = node.attr
    db = MappingDatabase(auto_load_learned=False)
    mapping = db.get_constant(lib, const)
    return mapping.cpp_value if mapping else None


def _create_function_call_operation(call_node, output_var: str,
                                    output_type: TypeHint,
                                    index: int) -> Optional[IROperation]:
    """Create function call operation from AST node."""
    unwrapped_call = _unwrap_type_cast(call_node, output_var, output_type, index)
    if unwrapped_call is not call_node:
        return unwrapped_call

    chained_call = _handle_method_chaining(call_node, output_var, output_type, index)
    if chained_call:
        return chained_call

    source_lib, function_name, source_object, op_type = _analyze_function_call_type(
        call_node.func
    )

    if op_type is None:
        return None

    args = _extract_function_args(call_node, function_name, source_lib)
    kwargs = _extract_function_kwargs(call_node)

    return _build_ir_operation(
        index, op_type, output_var, output_type, function_name,
        args, kwargs, source_lib, source_object
    )


def _handle_method_chaining(call_node, output_var: str, output_type: TypeHint,
                             index: int) -> Optional[IROperation]:
    """Handle chained method calls like a.method1().method2()."""
    if not isinstance(call_node.func, ast.Attribute):
        return None

    if not isinstance(call_node.func.value, ast.Call):
        return None

    chain_parts = []
    current = call_node

    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        method_name = current.func.attr
        args = [ast.unparse(arg) for arg in current.args]
        chain_parts.append((method_name, args))
        current = current.func.value

    if not chain_parts:
        return None

    base_obj = ast.unparse(current)
    chain_parts.reverse()

    chain_expr = base_obj
    for method, args in chain_parts:
        args_str = ", ".join(args)
        chain_expr = f"{chain_expr}.{method}({args_str})"

    return IROperation(
        id=f"op_{index + 1}",
        op_type=OperationType.ASSIGNMENT,
        output=output_var,
        output_type_hint=output_type,
        args=[chain_expr]
    )


def _extract_function_kwargs(call_node) -> dict:
    """Extract keyword arguments from function call."""
    return {kw.arg: ast.unparse(kw.value) for kw in call_node.keywords}


def _unwrap_type_cast(call_node, output_var: str, output_type: TypeHint,
                     index: int) -> Optional[IROperation]:
    """Unwrap type cast functions like int() or float()."""
    if isinstance(call_node.func, ast.Name) and call_node.func.id in ('int', 'float'):
        if len(call_node.args) == 1 and isinstance(call_node.args[0], ast.Call):
            return _create_function_call_operation(
                call_node.args[0], output_var, output_type, index
            )
    return call_node


def _extract_function_args(call_node, function_name: str, source_lib: str) -> list:
    """Extract and process function arguments."""
    args = []
    for arg in call_node.args:
        args.extend(_expand_tuple_args(arg))

    if function_name == 'cvtColor' and source_lib == 'cv2':
        args = args[:1] if args else []

    return args


def _build_ir_operation(index: int, op_type, output_var: str, output_type: TypeHint,
                       function_name: str, args: list, kwargs: dict,
                       source_lib: Optional[str], source_object: Optional[str]) -> IROperation:
    """Build IR operation from components."""
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
    for stmt in reversed(body):
        if isinstance(stmt, ast.Return) and stmt.value:
            if isinstance(stmt.value, ast.Name):
                var_name = stmt.value.id
                var_type = type_context.get(var_name, TypeHint('auto'))
                return [IROutput(var_name, var_type)]
            elif isinstance(stmt.value, ast.Tuple):
                outputs = []
                for elt in stmt.value.elts:
                    if isinstance(elt, ast.Name):
                        var_name = elt.id
                        var_type = type_context.get(var_name, TypeHint('auto'))
                        outputs.append(IROutput(var_name, var_type))
                return outputs

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


def process_pipeline_conversion(source_file: str, generator: CodeGenerator,
                                verbose: bool, test_input: Optional[str] = None,
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

    pre_pipeline = _build_preprocess_pipeline(separated, type_engine, ast_parser)
    inf_pipeline = _build_inference_pipeline(separated, type_engine, ast_parser)
    post_pipeline = _build_postprocess_pipeline(separated, type_engine, ast_parser)

    return pre_pipeline, inf_pipeline, post_pipeline


def _build_preprocess_pipeline(separated, type_engine, ast_parser) -> Optional[IRPipeline]:
    """Build preprocessing pipeline if available."""
    if not separated.preprocess:
        return None
    return _build_pipeline_from_segment(
        separated.preprocess, "preprocess", type_engine, ast_parser
    )


def _build_inference_pipeline(separated, type_engine, ast_parser) -> Optional[IRPipeline]:
    """Build inference pipeline if available."""
    if not separated.inference:
        return None

    inf_pipeline = _build_pipeline_from_segment(
        separated.inference, "inference", type_engine, ast_parser
    )
    inference_code = '\n'.join(ast.unparse(stmt) for stmt in separated.inference.body)
    inf_pipeline.metadata['inference_code'] = inference_code
    return inf_pipeline


def _build_postprocess_pipeline(separated, type_engine, ast_parser) -> Optional[IRPipeline]:
    """Build postprocessing pipeline if available."""
    if not separated.postprocess:
        return None
    return _build_pipeline_from_segment(
        separated.postprocess, "postprocess", type_engine, ast_parser
    )


def _extract_function_name(tree) -> Optional[str]:
    """Extract function name from AST."""
    for node in tree.body:
        if isinstance(node, __import__('ast').FunctionDef):
            return node.name
    return None


def _build_pipeline_from_segment(segment, name: str, type_engine, parser):
    """Build IR pipeline from segment."""
    temp_module = ast.Module(body=segment.body, type_ignores=[])
    type_context = {}

    inputs = [IRInput(var, TypeHint('auto')) for var in segment.inputs]
    operations = []

    for i, stmt in enumerate(segment.body):
        ops = _extract_operations_from_statement(stmt, type_context, i)
        operations.extend(ops)

    outputs = [IROutput(var, TypeHint('auto')) for var in segment.outputs]

    return IRPipeline(
        name=name,
        inputs=inputs,
        operations=operations,
        outputs=outputs,
        metadata={'source': 'pipeline_segment'}
    )


def validate_generated_code(project_dir: str, func_name: str, source_file: str,
                           test_input: str, verbose: bool, project_name: Optional[str] = None) -> None:
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
                       func_name: str, test_input: str, verbose: bool) -> Optional[float]:
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
                    executable_name: str, test_input: str) -> Optional[float]:
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
            return _handle_ir_only_mode(ast_parser, functions_to_convert, tree)

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
