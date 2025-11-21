"""Python AST Parser for extracting code structure information."""

import ast
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    alias: Optional[str] = None
    imported_names: List[str] = None

    def __post_init__(self):
        if self.imported_names is None:
            self.imported_names = []


@dataclass
class FunctionInfo:
    """Information about a function definition."""
    name: str
    args: List[str]
    returns: Optional[str]
    body: List[ast.stmt]
    decorators: List[str]
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
        self.imports: List[ImportInfo] = []
        self.functions: List[FunctionInfo] = []
        self.global_assignments: Dict[str, ast.AST] = {}
        self._current_function: Optional[str] = None

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
            raise SyntaxError(f"Failed to parse Python code: {e}")

    def _analyze_tree(self, tree: ast.Module):
        """Analyze AST tree and extract information."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._extract_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._extract_import_from(node)
            elif isinstance(node, ast.FunctionDef):
                self._extract_function(node)

    def _extract_import(self, node: ast.Import):
        """Extract information from import statement."""
        for alias in node.names:
            self.imports.append(ImportInfo(
                module=alias.name,
                alias=alias.asname
            ))

    def _extract_import_from(self, node: ast.ImportFrom):
        """Extract information from 'from X import Y' statement."""
        module = node.module or ''
        imported_names = [alias.name for alias in node.names]
        self.imports.append(ImportInfo(
            module=module,
            imported_names=imported_names
        ))

    def _extract_function(self, node: ast.FunctionDef):
        """Extract information from function definition."""
        args = [arg.arg for arg in node.args.args]
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]

        returns = ast.unparse(node.returns) if node.returns else None

        self.functions.append(FunctionInfo(
            name=node.name,
            args=args,
            returns=returns,
            body=node.body,
            decorators=decorators,
            lineno=node.lineno
        ))

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        return ast.unparse(decorator)

    def get_library_usage(self) -> Set[str]:
        """
        Get set of all imported libraries.

        Returns:
            Set of library names used in the code
        """
        libraries = set()
        for imp in self.imports:
            base_lib = imp.module.split('.')[0]
            libraries.add(base_lib)
        return libraries

    def get_function_calls(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """
        Extract all function calls from the AST.

        Args:
            tree: AST Module node

        Returns:
            List of dictionaries containing function call information
        """
        calls = []

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                call_info = {
                    'func': ast.unparse(node.func),
                    'args': [ast.unparse(arg) for arg in node.args],
                    'kwargs': {kw.arg: ast.unparse(kw.value) for kw in node.keywords},
                    'lineno': node.lineno
                }
                calls.append(call_info)
                self.generic_visit(node)

        visitor = CallVisitor()
        visitor.visit(tree)
        return calls

    def get_assignments(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """
        Extract all assignment statements from the AST.

        Args:
            tree: AST Module node

        Returns:
            List of dictionaries containing assignment information
        """
        assignments = []

        class AssignVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                for target in node.targets:
                    assign_info = {
                        'target': ast.unparse(target),
                        'value': ast.unparse(node.value),
                        'lineno': node.lineno
                    }
                    assignments.append(assign_info)
                self.generic_visit(node)

            def visit_AnnAssign(self, node):
                assign_info = {
                    'target': ast.unparse(node.target),
                    'value': ast.unparse(node.value) if node.value else None,
                    'annotation': ast.unparse(node.annotation),
                    'lineno': node.lineno
                }
                assignments.append(assign_info)
                self.generic_visit(node)

        visitor = AssignVisitor()
        visitor.visit(tree)
        return assignments
