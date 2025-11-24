"""Python AST Parser for extracting code structure information."""

import ast
from dataclasses import dataclass
from typing import Any


@dataclass
class ImportInfo:
    """Information about an import statement."""

    module: str
    alias: str | None = None
    imported_names: list[str] | None = None

    def __post_init__(self) -> None:
        if self.imported_names is None:
            self.imported_names = []


@dataclass
class FunctionInfo:
    """Information about a function definition."""

    name: str
    args: list[str]
    returns: str | None
    body: list[ast.stmt]
    decorators: list[str]
    lineno: int


class PythonASTParser:
    """
    Parses Python source code using AST module.

    This parser extracts:
    - Import statements
    - Function definitions
    - Variable assignments
    - Function calls
    - Control flow structures
    """

    def __init__(self):
        self.imports: list[ImportInfo] = []
        self.functions: list[FunctionInfo] = []
        self.global_assignments: dict[str, ast.AST] = {}

    def parse(self, source_code: str) -> ast.Module:
        """
        Parse Python source code into AST.

        Args:
            source_code: Python source code as string

        Returns:
            AST Module node

        Raises:
            SyntaxError: If the source code is invalid
        """
        try:
            tree = ast.parse(source_code)
            self._analyze_tree(tree)
            return tree
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse Python code: {e}") from e

    def _analyze_tree(self, tree: ast.Module) -> None:
        """Analyze AST tree and extract information."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._extract_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._extract_import_from(node)
            elif isinstance(node, ast.FunctionDef):
                self._extract_function(node)

    def _extract_import(self, node: ast.Import) -> None:
        """Extract information from import statement."""
        for alias in node.names:
            self.imports.append(ImportInfo(module=alias.name, alias=alias.asname))

    def _extract_import_from(self, node: ast.ImportFrom) -> None:
        """Extract information from 'from X import Y' statement."""
        module = node.module or ""
        imported_names = [alias.name for alias in node.names]
        self.imports.append(ImportInfo(module=module, imported_names=imported_names))

    def _extract_function(self, node: ast.FunctionDef) -> None:
        """Extract information from function definition."""
        args = [arg.arg for arg in node.args.args]
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]

        returns = ast.unparse(node.returns) if node.returns else None

        self.functions.append(
            FunctionInfo(
                name=node.name,
                args=args,
                returns=returns,
                body=node.body,
                decorators=decorators,
                lineno=node.lineno,
            )
        )

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        return ast.unparse(decorator)

    def get_library_usage(self) -> set[str]:
        """
        Get set of all imported libraries.

        Returns:
            Set of library names used in the code
        """
        libraries = set()
        for imp in self.imports:
            base_lib = imp.module.split(".")[0]
            libraries.add(base_lib)
        return libraries

    def get_function_calls(self, tree: ast.Module) -> list[dict[str, str | list | int]]:
        """
        Extract all function calls from the AST.

        Args:
            tree: AST Module node

        Returns:
            List of dictionaries containing function call information
        """
        calls: list[dict[str, str | list | int]] = []

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                call_info: dict[str, str | list | int] = {
                    "func": ast.unparse(node.func),
                    "args": [ast.unparse(arg) for arg in node.args],
                    "kwargs": {kw.arg: ast.unparse(kw.value) for kw in node.keywords},
                    "lineno": node.lineno,
                }
                calls.append(call_info)
                self.generic_visit(node)

        visitor = CallVisitor()
        visitor.visit(tree)
        return calls

    def get_assignments(self, tree: ast.Module) -> list[dict[str, str | int | None]]:
        """
        Extract all assignment statements from the AST.

        Args:
            tree: AST Module node

        Returns:
            List of dictionaries containing assignment information
        """
        assignments: list[dict[str, str | int | None]] = []

        class AssignVisitor(ast.NodeVisitor):
            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    assign_info: dict[str, str | int] = {
                        "target": ast.unparse(target),
                        "value": ast.unparse(node.value),
                        "lineno": node.lineno,
                    }
                    assignments.append(assign_info)
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                assign_info: dict[str, str | int | None] = {
                    "target": ast.unparse(node.target),
                    "value": ast.unparse(node.value) if node.value else None,
                    "annotation": ast.unparse(node.annotation),
                    "lineno": node.lineno,
                }
                assignments.append(assign_info)
                self.generic_visit(node)

        visitor = AssignVisitor()
        visitor.visit(tree)
        return assignments

    def get_main_block_functions(self, tree: ast.Module) -> list[str]:
        """
        Extract function names called in 'if __name__ == "__main__":' block.

        Args:
            tree: AST Module node

        Returns:
            List of function names called in main block
        """
        main_functions = []

        for node in tree.body:
            if isinstance(node, ast.If):
                # Check if this is the __name__ == "__main__" condition
                if self._is_main_condition(node.test):
                    # Extract function calls from the if block
                    main_functions = self._extract_function_calls_from_block(node.body)
                    break

        return main_functions

    def get_main_block_body(self, tree: ast.Module) -> list[ast.stmt] | None:
        """
        Extract the body of 'if __name__ == "__main__":' block.

        Args:
            tree: AST Module node

        Returns:
            List of statements in main block, or None if not found
        """
        for node in tree.body:
            if isinstance(node, ast.If):
                if self._is_main_condition(node.test):
                    return node.body

        return None

    def _is_main_condition(self, test_node: ast.expr) -> bool:
        """Check if test node is __name__ == "__main__" condition."""
        if isinstance(test_node, ast.Compare):
            left = test_node.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                if len(test_node.ops) == 1 and isinstance(test_node.ops[0], ast.Eq):
                    if len(test_node.comparators) == 1:
                        comp = test_node.comparators[0]
                        if isinstance(comp, ast.Constant) and comp.value == "__main__":
                            return True
        return False

    def _extract_function_calls_from_block(self, block: list[ast.stmt]) -> list[str]:
        """Extract function names from a block of statements."""
        function_calls = []

        class FunctionCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.calls = []

            def visit_Call(self, node: ast.Call):
                # Extract simple function calls (not methods)
                if isinstance(node.func, ast.Name):
                    self.calls.append(node.func.id)
                self.generic_visit(node)

        visitor = FunctionCallVisitor()
        for stmt in block:
            visitor.visit(stmt)

        return visitor.calls
