"""IR Builder for converting Python AST to Intermediate Representation."""

import ast

from core.analysis.inferencer import TypeInferenceEngine
from core.analysis.parser import FunctionInfo, PythonASTParser
from core.intermediate.schema import (
    IRInput,
    IROperation,
    IROutput,
    IRPipeline,
    OperationType,
    TypeHint,
)
from core.mapping.database import MappingDatabase


class IRBuilder:
    """
    Builder class for constructing IR pipelines from Python AST.
    """

    def build_pipeline(
        self,
        parser: PythonASTParser,
        func_name: str,
        tree: ast.Module,
        type_engine: TypeInferenceEngine,
    ) -> IRPipeline:
        """Build IR pipeline from parsed function."""
        func_info = self._find_function_info(parser, func_name)
        type_context = self._build_type_context(tree, func_name, type_engine)

        inputs = self._build_pipeline_inputs(func_info, type_context)
        operations = self._build_pipeline_operations(func_info, type_context)
        outputs = self._determine_outputs(func_info.body, type_context)
        libraries = list(parser.get_library_usage())

        return IRPipeline(
            name=func_name,
            inputs=inputs,
            operations=operations,
            outputs=outputs,
            metadata={"source_file": parser.__class__.__name__, "libraries": libraries},
        )

    def _find_function_info(
        self, parser: PythonASTParser, func_name: str
    ) -> FunctionInfo:
        """Find function information from parser."""
        func_info = next((f for f in parser.functions if f.name == func_name), None)
        if not func_info:
            raise ValueError(f"Function not found: {func_name}")
        return func_info

    def _build_type_context(
        self, tree: ast.Module, func_name: str, type_engine: TypeInferenceEngine
    ) -> dict[str, TypeHint]:
        """Build type context for function."""
        func_node = next(
            (
                node
                for node in tree.body
                if hasattr(node, "name") and node.name == func_name
            ),
            None,
        )
        if not func_node:
            return {}
        return type_engine.analyze_function(func_node)

    def _build_pipeline_inputs(
        self, func_info: FunctionInfo, type_context: dict[str, TypeHint]
    ) -> list[IRInput]:
        """Build IR inputs from function arguments."""
        return [
            IRInput(arg, type_context.get(arg, TypeHint("auto")))
            for arg in func_info.args
        ]

    def _build_pipeline_operations(
        self, func_info: FunctionInfo, type_context: dict[str, TypeHint]
    ) -> list[IROperation]:
        """Build IR operations from function body."""
        operations = []
        for i, stmt in enumerate(func_info.body):
            ops = self._extract_operations_from_statement(stmt, type_context, i)
            operations.extend(ops)
        return operations

    def _extract_operations_from_statement(
        self, stmt: ast.stmt, type_context: dict[str, TypeHint], index: int
    ) -> list[IROperation]:
        """Extract IR operations from statement."""
        operations = []

        if isinstance(stmt, ast.Assign):
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                output_var = target.id
                output_type = type_context.get(output_var, TypeHint("auto"))
                op = self._create_assignment_operation(
                    stmt.value, output_var, output_type, index
                )
                if op:
                    operations.append(op)
        elif isinstance(stmt, ast.If):
            op = self._create_conditional_operation(stmt, type_context, index)
            if op:
                operations.append(op)
        elif isinstance(stmt, (ast.For, ast.While)):
            op = self._create_loop_operation(stmt, type_context, index)
            if op:
                operations.append(op)

        return operations

    def _create_assignment_operation(
        self, value_node: ast.expr, output_var: str, output_type: TypeHint, index: int
    ) -> IROperation | None:
        """Create operation from assignment value node."""
        if isinstance(value_node, ast.Call):
            return self._create_function_call_operation(
                value_node, output_var, output_type, index
            )
        elif isinstance(value_node, ast.BinOp):
            return self._create_binary_operation(
                value_node, output_var, output_type, index
            )
        elif isinstance(value_node, ast.List):
            return self._create_list_assignment(
                value_node, output_var, output_type, index
            )
        elif isinstance(value_node, ast.Subscript):
            return self._create_subscript_assignment(
                value_node, output_var, output_type, index
            )
        elif isinstance(value_node, ast.Name):
            return self._create_variable_assignment(
                value_node, output_var, output_type, index
            )
        elif isinstance(value_node, ast.Constant):
            return self._create_constant_assignment(
                value_node, output_var, output_type, index
            )
        return None

    def _create_list_assignment(
        self, list_node: ast.List, output_var: str, output_type: TypeHint, index: int
    ) -> IROperation:
        """Create C++ vector assignment from Python list."""
        elements = self._convert_list_elements(list_node)
        cpp_vector = "std::vector<std::string>{" + ", ".join(elements) + "}"
        return IROperation(
            id=f"op_{index + 1}",
            op_type=OperationType.ASSIGNMENT,
            output=output_var,
            output_type_hint=output_type,
            args=[cpp_vector],
        )

    def _convert_list_elements(self, list_node: ast.List) -> list[str]:
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

    def _create_subscript_assignment(
        self,
        subscript_node: ast.Subscript,
        output_var: str,
        output_type: TypeHint,
        index: int,
    ) -> IROperation:
        """Create array subscript assignment operation."""
        return self._create_simple_assignment(
            output_var, output_type, index, ast.unparse(subscript_node)
        )

    def _create_variable_assignment(
        self, name_node: ast.Name, output_var: str, output_type: TypeHint, index: int
    ) -> IROperation:
        """Create simple variable assignment operation."""
        return self._create_simple_assignment(
            output_var, output_type, index, name_node.id
        )

    def _create_constant_assignment(
        self,
        const_node: ast.Constant,
        output_var: str,
        output_type: TypeHint,
        index: int,
    ) -> IROperation:
        """Create constant value assignment operation."""
        cpp_value = self._convert_constant_to_cpp(const_node.value)
        return self._create_simple_assignment(output_var, output_type, index, cpp_value)

    def _create_simple_assignment(
        self, output_var: str, output_type: TypeHint, index: int, value: str
    ) -> IROperation:
        """Create simple assignment IR operation."""
        return IROperation(
            id=f"op_{index + 1}",
            op_type=OperationType.ASSIGNMENT,
            output=output_var,
            output_type_hint=output_type,
            args=[value],
        )

    def _create_conditional_operation(
        self, if_stmt: ast.If, type_context: dict[str, TypeHint], index: int
    ) -> IROperation:
        """Create conditional (if/else) operation."""
        condition = ast.unparse(if_stmt.test)

        true_ops = []
        for i, stmt in enumerate(if_stmt.body):
            ops = self._extract_operations_from_statement(
                stmt, type_context, index * 1000 + i
            )
            true_ops.extend(ops)

        false_ops = []
        for i, stmt in enumerate(if_stmt.orelse):
            ops = self._extract_operations_from_statement(
                stmt, type_context, index * 1000 + 100 + i
            )
            false_ops.extend(ops)

        return IROperation(
            id=f"op_{index + 1}",
            op_type=OperationType.CONDITIONAL,
            output="",
            output_type_hint=TypeHint("void"),
            condition=condition,
            true_branch=true_ops,
            false_branch=false_ops,
        )

    def _create_loop_operation(
        self, loop_stmt: ast.For | ast.While, type_context: dict[str, TypeHint], index: int
    ) -> IROperation:
        """Create loop (for/while) operation."""
        if isinstance(loop_stmt, ast.For):
            loop_var = (
                loop_stmt.target.id if isinstance(loop_stmt.target, ast.Name) else ""
            )
            iterable = ast.unparse(loop_stmt.iter)
        else:
            loop_var = ""
            iterable = ast.unparse(loop_stmt.test)

        loop_ops = []
        for i, stmt in enumerate(loop_stmt.body):
            ops = self._extract_operations_from_statement(
                stmt, type_context, index * 1000 + i
            )
            loop_ops.extend(ops)

        return IROperation(
            id=f"op_{index + 1}",
            op_type=OperationType.LOOP,
            output="",
            output_type_hint=TypeHint("void"),
            loop_var=loop_var,
            iterable=iterable,
            loop_body=loop_ops,
        )

    def _convert_constant_to_cpp(self, value: str | bool | int | float) -> str:
        """Convert Python constant to C++ representation."""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "true" if value else "false"
        else:
            return str(value)

    def _create_binary_operation(
        self, binop_node: ast.BinOp, output_var: str, output_type: TypeHint, index: int
    ) -> IROperation | None:
        """Create binary operation from AST node."""
        left = self._process_operand(binop_node.left)
        right = self._process_operand(binop_node.right)
        operator = self._get_operator_symbol(binop_node.op)

        return IROperation(
            id=f"op_{index + 1}",
            op_type=OperationType.ARITHMETIC,
            output=output_var,
            output_type_hint=output_type,
            operands=[left, right],
            operator=operator,
        )

    def _process_operand(self, node: ast.expr) -> str:
        """Process operand, unwrapping astype() calls."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "astype"
        ):
            return ast.unparse(node.func.value)
        return ast.unparse(node)

    def _get_operator_symbol(self, op_node: ast.operator) -> str:
        """Convert AST operator to symbol."""
        op_map: dict[type, str] = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
        }
        return op_map.get(type(op_node), "?")

    def _analyze_function_call_type(
        self, func: ast.expr
    ) -> tuple[str | None, str | None, str | None, OperationType | None]:
        """Analyze function call type and extract metadata."""
        known_libraries = {"cv2", "np", "numpy", "torch", "tf", "PIL", "Image"}

        if isinstance(func, ast.Attribute):
            return self._analyze_attribute_call(func, known_libraries)
        elif isinstance(func, ast.Name):
            return None, func.id, None, OperationType.FUNCTION_CALL
        else:
            return None, None, None, None

    def _analyze_attribute_call(
        self, func: ast.Attribute, known_libraries: set
    ) -> tuple[str | None, str | None, str | None, OperationType | None]:
        """Analyze attribute function call."""
        if isinstance(func.value, ast.Name):
            lib_or_var = func.value.id
            if lib_or_var in known_libraries:
                return lib_or_var, func.attr, lib_or_var, OperationType.FUNCTION_CALL
            return None, func.attr, lib_or_var, OperationType.METHOD_CALL

        source_object = ast.unparse(func.value)
        return None, func.attr, source_object, OperationType.METHOD_CALL

    def _expand_tuple_args(self, arg_node: ast.expr) -> list[str]:
        """Expand tuple arguments into individual elements."""
        if isinstance(arg_node, ast.Tuple):
            return [self._convert_constant(elt) for elt in arg_node.elts]
        return [self._convert_constant(arg_node)]

    def _convert_constant(self, node: ast.expr) -> str:
        """Convert Python constant to C++ value if it's a known constant."""
        if isinstance(node, ast.Attribute):
            mapped_value = self._try_map_attribute_constant(node)
            if mapped_value:
                return mapped_value

        return ast.unparse(node)

    def _try_map_attribute_constant(self, node: ast.Attribute) -> str | None:
        """Try to map attribute constant to C++ equivalent."""
        if not isinstance(node.value, ast.Name):
            return None

        lib = node.value.id
        const = node.attr
        db = MappingDatabase(auto_load_learned=False)
        mapping = db.get_constant(lib, const)
        return mapping.cpp_value if mapping else None

    def _create_function_call_operation(
        self, call_node: ast.Call, output_var: str, output_type: TypeHint, index: int
    ) -> IROperation | None:
        """Create function call operation from AST node."""
        unwrapped_call = self._unwrap_type_cast(
            call_node, output_var, output_type, index
        )
        if unwrapped_call:
            return unwrapped_call

        chained_call = self._handle_method_chaining(
            call_node, output_var, output_type, index
        )
        if chained_call:
            return chained_call

        source_lib, function_name, source_object, op_type = (
            self._analyze_function_call_type(call_node.func)
        )

        if op_type is None:
            return None

        args = self._extract_function_args(call_node, function_name, source_lib)
        kwargs = self._extract_function_kwargs(call_node)

        return self._build_ir_operation(
            index,
            op_type,
            output_var,
            output_type,
            function_name,
            args,
            kwargs,
            source_lib,
            source_object,
        )

    def _handle_method_chaining(
        self, call_node: ast.Call, output_var: str, output_type: TypeHint, index: int
    ) -> IROperation | None:
        """Handle chained method calls like a.method1().method2()."""
        if not isinstance(call_node.func, ast.Attribute):
            return None

        if not isinstance(call_node.func.value, ast.Call):
            return None

        chain_parts: list[tuple[str, list[str]]] = []
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
            args=[chain_expr],
        )

    def _extract_function_kwargs(self, call_node: ast.Call) -> dict[str, str]:
        """Extract keyword arguments from function call."""
        return {kw.arg: ast.unparse(kw.value) for kw in call_node.keywords}

    def _unwrap_type_cast(
        self, call_node: ast.Call, output_var: str, output_type: TypeHint, index: int
    ) -> IROperation | None:
        """Unwrap type cast functions like int() or float()."""
        if (
            isinstance(call_node.func, ast.Name)
            and call_node.func.id in ("int", "float")
            and len(call_node.args) == 1
            and isinstance(call_node.args[0], ast.Call)
        ):
            return self._create_function_call_operation(
                call_node.args[0], output_var, output_type, index
            )
        return None

    def _extract_function_args(
        self, call_node: ast.Call, function_name: str | None, source_lib: str | None
    ) -> list[str]:
        """Extract and process function arguments."""
        args: list[str] = []
        for arg in call_node.args:
            args.extend(self._expand_tuple_args(arg))

        return args

    def _build_ir_operation(
        self,
        index: int,
        op_type: OperationType,
        output_var: str,
        output_type: TypeHint,
        function_name: str | None,
        args: list[str],
        kwargs: dict[str, str],
        source_lib: str | None,
        source_object: str | None,
    ) -> IROperation:
        """Build IR operation from components."""
        op_dict = {
            "id": f"op_{index + 1}",
            "op_type": op_type,
            "output": output_var,
            "output_type_hint": output_type,
            "function": function_name,
            "args": args,
            "kwargs": kwargs,
        }

        if source_lib:
            op_dict["source_lib"] = source_lib
        if source_object:
            op_dict["source_object"] = source_object

        return IROperation(**op_dict)

    def _determine_outputs(
        self, body: list[ast.stmt], type_context: dict[str, TypeHint]
    ) -> list[IROutput]:
        """Determine output variables from function body."""
        for stmt in reversed(body):
            if isinstance(stmt, ast.Return) and stmt.value:
                if isinstance(stmt.value, ast.Name):
                    var_name = stmt.value.id
                    var_type = type_context.get(var_name, TypeHint("auto"))
                    return [IROutput(var_name, var_type)]
                elif isinstance(stmt.value, ast.Tuple):
                    outputs = []
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Name):
                            var_name = elt.id
                            var_type = type_context.get(var_name, TypeHint("auto"))
                            outputs.append(IROutput(var_name, var_type))
                    return outputs

    def build_main_block_operations(
        self, main_body: list[ast.stmt], type_engine: TypeInferenceEngine
    ) -> list[IROperation]:
        """
        Build IR operations from __main__ block statements.

        Args:
            main_body: List of statements from __main__ block
            type_engine: Type inference engine

        Returns:
            List of IR operations
        """
        # Build a simple type context for main block
        type_context = {}
        for stmt in main_body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        # Infer type from assignment
                        var_type = self._infer_type_from_value(stmt.value)
                        type_context[target.id] = var_type

        operations = []
        for i, stmt in enumerate(main_body):
            ops = self._extract_main_block_statement(stmt, type_context, i)
            operations.extend(ops)

        return operations

    def _extract_main_block_statement(
        self, stmt: ast.stmt, type_context: dict, index: int
    ) -> list[IROperation]:
        """Extract operations from a single statement in __main__ block."""
        operations = []

        if isinstance(stmt, ast.Assign):
            # Handle assignments: a = 1, img = cv2.imread(...)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    # Check if it's a function call
                    if isinstance(stmt.value, ast.Call):
                        ops = self._extract_operations_from_statement(
                            stmt, type_context, index
                        )
                        operations.extend(ops)
                    else:
                        # Simple assignment: a = 1, s = "hello"
                        value_str = ast.unparse(stmt.value)

                        # Skip assignments that are argument processing (sys.argv, len(sys.argv), etc.)
                        if 'sys.argv' in value_str or var_name == 'image_path':
                            continue

                        var_type = type_context.get(var_name, TypeHint("auto"))
                        op = IROperation(
                            id=f"main_assign_{index}",
                            op_type=OperationType.ASSIGNMENT,
                            output=var_name,
                            output_type_hint=var_type,
                            args=[value_str],
                        )
                        operations.append(op)
        elif isinstance(stmt, ast.Expr):
            # Handle expression statements (function calls, print, etc)
            if isinstance(stmt.value, ast.Call):
                ops = self._extract_operations_from_statement(stmt, type_context, index)
                operations.extend(ops)
        elif isinstance(stmt, ast.If):
            # Handle if statements
            ops = self._extract_operations_from_statement(stmt, type_context, index)
            operations.extend(ops)
        else:
            # Try to extract with existing method
            ops = self._extract_operations_from_statement(stmt, type_context, index)
            operations.extend(ops)

        return operations

    def _infer_type_from_value(self, value_node: ast.expr) -> TypeHint:
        """Infer type hint from assignment value."""
        if isinstance(value_node, ast.Call):
            # Try to infer from function call
            if isinstance(value_node.func, ast.Attribute):
                func_name = value_node.func.attr
                if func_name in ("imread", "resize", "cvtColor"):
                    return TypeHint("img::Image")
            elif isinstance(value_node.func, ast.Name):
                func_name = value_node.func.id
                # If it's calling our defined functions, check their return types
                # For now, assume they return ImageF
                return TypeHint("img::ImageF")

        # Default type
        return TypeHint("auto")
