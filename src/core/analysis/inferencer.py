"""Type Inference Engine for static type analysis."""

import ast

from core.intermediate.schema import TypeHint


class TypeInferenceEngine:
    """
    Infers types for Python variables using static analysis.

    This engine uses:
    1. Type annotations (if present)
    2. Literal value analysis
    3. Function return type tracking
    4. Library-specific knowledge (e.g., cv2.imread returns ndarray)
    """

    def __init__(self):
        self.type_map: dict[str, TypeHint] = {}
        self.library_signatures = self._initialize_library_signatures()

    def _initialize_library_signatures(self) -> dict[str, dict[str, TypeHint]]:
        """
        Initialize known function signatures for common libraries.

        Returns:
            Dictionary mapping library.function to return type
        """
        return {
            # OpenCV (cv2)
            "cv2.imread": TypeHint("numpy.ndarray", dtype="uint8", shape="(H, W, C)"),
            "cv2.resize": TypeHint("numpy.ndarray", dtype="same", shape="(H2, W2, C)"),
            "cv2.cvtColor": TypeHint("numpy.ndarray", dtype="same", shape="same"),
            "cv2.GaussianBlur": TypeHint("numpy.ndarray", dtype="same", shape="same"),
            # NumPy
            "numpy.array": TypeHint("numpy.ndarray"),
            "numpy.zeros": TypeHint("numpy.ndarray"),
            "numpy.ones": TypeHint("numpy.ndarray"),
            "numpy.reshape": TypeHint("numpy.ndarray", dtype="same", shape="new"),
            "numpy.transpose": TypeHint("numpy.ndarray", dtype="same"),
            "numpy.mean": TypeHint("numpy.ndarray", dtype="float64"),
            "numpy.std": TypeHint("numpy.ndarray", dtype="float64"),
            # Pillow (PIL)
            "PIL.Image.open": TypeHint("PIL.Image.Image"),
            # Librosa
            "librosa.load": TypeHint("tuple[numpy.ndarray, int]"),
            "librosa.stft": TypeHint("numpy.ndarray", dtype="complex64"),
            "librosa.feature.melspectrogram": TypeHint(
                "numpy.ndarray", dtype="float32"
            ),
        }

    def infer_type(
        self, node: ast.AST, context: dict[str, TypeHint]
    ) -> TypeHint | None:
        """
        Infer the type of an AST node.

        Args:
            node: AST node to analyze
            context: Current variable type context

        Returns:
            TypeHint if type can be inferred, None otherwise
        """
        if isinstance(node, ast.Constant):
            return self._infer_constant_type(node)
        elif isinstance(node, ast.Name):
            return context.get(node.id)
        elif isinstance(node, ast.Call):
            return self._infer_call_type(node, context)
        elif isinstance(node, ast.BinOp):
            return self._infer_binop_type(node, context)
        elif isinstance(node, ast.Subscript):
            return self._infer_subscript_type(node, context)
        elif isinstance(node, ast.Attribute):
            return self._infer_attribute_type(node, context)
        else:
            return None

    def _infer_constant_type(self, node: ast.Constant) -> TypeHint:
        """Infer type from constant literal"""
        value = node.value
        if isinstance(value, int):
            return TypeHint("int", is_const=True)
        elif isinstance(value, float):
            return TypeHint("float", is_const=True)
        elif isinstance(value, str):
            return TypeHint("str", is_const=True)
        elif isinstance(value, bool):
            return TypeHint("bool", is_const=True)
        else:
            return TypeHint("unknown", is_const=True)

    def _infer_call_type(
        self, node: ast.Call, context: dict[str, TypeHint]
    ) -> TypeHint | None:
        """Infer return type of a function call"""
        func_name = self._get_full_function_name(node.func, context)

        # Check library signatures
        if func_name in self.library_signatures:
            return self.library_signatures[func_name]

        # Check for method calls on known types
        if isinstance(node.func, ast.Attribute):
            obj_type = self.infer_type(node.func.value, context)
            if obj_type:
                method_sig = f"{obj_type.base_type}.{node.func.attr}"
                if method_sig in self.library_signatures:
                    return self.library_signatures[method_sig]

                # NumPy array methods
                if obj_type.base_type == "numpy.ndarray":
                    return self._infer_ndarray_method(node.func.attr, obj_type)

        return None

    def _infer_ndarray_method(
        self, method: str, array_type: TypeHint
    ) -> TypeHint | None:
        """Infer return type of NumPy array methods"""
        if method == "astype":
            # Returns same array with different dtype
            return TypeHint("numpy.ndarray", shape=array_type.shape)
        elif method == "reshape" or method == "transpose":
            return TypeHint("numpy.ndarray", dtype=array_type.dtype)
        elif method in ["mean", "sum", "std", "var"]:
            return TypeHint("numpy.ndarray", dtype="float64")
        elif method == "copy":
            return array_type
        else:
            return TypeHint("numpy.ndarray")

    def _infer_binop_type(
        self, node: ast.BinOp, context: dict[str, TypeHint]
    ) -> TypeHint | None:
        """Infer type of binary operation"""
        left_type = self.infer_type(node.left, context)
        right_type = self.infer_type(node.right, context)

        if not left_type:
            return right_type
        if not right_type:
            return left_type

        # NumPy array operations
        if left_type.base_type == "numpy.ndarray":
            return left_type

        # Numeric operations
        if left_type.base_type in ["int", "float"] and right_type.base_type in [
            "int",
            "float",
        ]:
            if "float" in [left_type.base_type, right_type.base_type]:
                return TypeHint("float")
            else:
                return TypeHint("int")

        return None

    def _infer_subscript_type(
        self, node: ast.Subscript, context: dict[str, TypeHint]
    ) -> TypeHint | None:
        """Infer type of subscript operation (e.g., array[0])"""
        value_type = self.infer_type(node.value, context)

        if value_type and value_type.base_type == "numpy.ndarray":
            # Subscripting an array typically returns an array of reduced dimension
            return TypeHint("numpy.ndarray", dtype=value_type.dtype)

        return None

    def _infer_attribute_type(
        self, node: ast.Attribute, context: dict[str, TypeHint]
    ) -> TypeHint | None:
        """Infer type of attribute access"""
        obj_type = self.infer_type(node.value, context)

        if obj_type and obj_type.base_type == "numpy.ndarray":
            if node.attr == "shape":
                return TypeHint("tuple[int, ...]")
            elif node.attr == "dtype":
                return TypeHint("numpy.dtype")

        return None

    def _get_full_function_name(
        self, func_node: ast.AST, context: dict[str, TypeHint]
    ) -> str:
        """
        Get the full qualified name of a function call.

        Args:
            func_node: Function AST node
            context: Variable type context

        Returns:
            Full function name (e.g., 'cv2.imread')
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            if isinstance(func_node.value, ast.Name):
                return f"{func_node.value.id}.{func_node.attr}"
            else:
                # Nested attributes
                base = self._get_full_function_name(func_node.value, context)
                return f"{base}.{func_node.attr}"
        else:
            return ast.unparse(func_node)

    def analyze_function(self, func_def: ast.FunctionDef) -> dict[str, TypeHint]:
        """
        Analyze a function definition and build type context.

        Args:
            func_def: Function definition AST node

        Returns:
            Dictionary mapping variable names to their inferred types
        """
        context: dict[str, TypeHint] = {}

        # Process function arguments with type annotations
        for arg in func_def.args.args:
            if arg.annotation:
                type_str = ast.unparse(arg.annotation)
                context[arg.arg] = self._parse_type_annotation(type_str)

        # Process function body
        for stmt in func_def.body:
            self._analyze_statement(stmt, context)

        return context

    def _analyze_statement(self, stmt: ast.stmt, context: dict[str, TypeHint]):
        """Analyze a statement and update type context"""
        if isinstance(stmt, ast.Assign):
            # Infer type from value
            value_type = self.infer_type(stmt.value, context)
            if value_type:
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        context[target.id] = value_type

        elif isinstance(stmt, ast.AnnAssign):
            # Use annotation if present
            if stmt.annotation:
                type_str = ast.unparse(stmt.annotation)
                type_hint = self._parse_type_annotation(type_str)
                if isinstance(stmt.target, ast.Name):
                    context[stmt.target.id] = type_hint

    def _parse_type_annotation(self, type_str: str) -> TypeHint:
        """Parse a type annotation string into TypeHint"""
        # Simplified parsing - could be extended
        if "ndarray" in type_str or "np.ndarray" in type_str:
            return TypeHint("numpy.ndarray")
        elif type_str == "str":
            return TypeHint("str")
        elif type_str == "int":
            return TypeHint("int")
        elif type_str == "float":
            return TypeHint("float")
        else:
            return TypeHint(type_str)
