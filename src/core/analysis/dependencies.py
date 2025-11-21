"""Recursive dependency resolution for multi-file Python projects."""

import ast
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImportInfo:
    """Information about a Python import statement."""
    module_name: str
    imported_names: list[str]
    is_local: bool
    file_path: Optional[Path]


@dataclass
class DependencyNode:
    """Node in dependency graph."""
    file_path: Path
    module_name: str
    imports: list[ImportInfo]
    ast_tree: ast.Module
    source_code: str


class DependencyResolver:
    """Recursively resolve and parse local imports."""

    STDLIB_MODULES = {
        'sys', 'os', 'pathlib', 'typing', 'dataclasses', 'collections',
        'itertools', 'functools', 're', 'json', 'math', 'random', 'time'
    }

    THIRD_PARTY_MODULES = {
        'numpy', 'cv2', 'PIL', 'scipy', 'pandas', 'sklearn',
        'torch', 'tensorflow', 'jax', 'matplotlib'
    }

    def __init__(self, root_dir: Optional[Path] = None):
        """
        Initialize dependency resolver.

        Args:
            root_dir: Root directory for resolving relative imports
        """
        self.root_dir = root_dir or Path.cwd()
        self.resolved: dict[str, DependencyNode] = {}
        self.visited: set[str] = set()

    def resolve(self, entry_file: Path) -> list[DependencyNode]:
        """
        Resolve all dependencies starting from entry file.

        Args:
            entry_file: Starting Python file

        Returns:
            List of DependencyNodes in topological order (dependencies first)
        """
        self.resolved.clear()
        self.visited.clear()

        self._resolve_recursive(entry_file)

        return self._topological_sort()

    def _resolve_recursive(self, file_path: Path) -> Optional[DependencyNode]:
        """Recursively resolve dependencies for a file."""
        if not file_path.exists():
            return None

        module_name = self._file_to_module_name(file_path)

        if module_name in self.visited:
            return self.resolved.get(module_name)

        self.visited.add(module_name)

        with open(file_path) as f:
            source_code = f.read()

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return None

        imports = self._extract_imports(tree, file_path)

        node = DependencyNode(
            file_path=file_path,
            module_name=module_name,
            imports=imports,
            ast_tree=tree,
            source_code=source_code
        )

        self.resolved[module_name] = node

        for imp in imports:
            if imp.is_local and imp.file_path:
                self._resolve_recursive(imp.file_path)

        return node

    def _extract_imports(self, tree: ast.Module, current_file: Path) -> list[ImportInfo]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(self._process_import(alias.name, [alias.asname or alias.name], current_file))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported = [alias.name for alias in node.names]
                    imports.append(self._process_import(node.module, imported, current_file, node.level))

        return imports

    def _process_import(self, module_name: str, imported_names: list[str],
                       current_file: Path, level: int = 0) -> ImportInfo:
        """Process import and determine if it's local."""
        is_local = self._is_local_module(module_name)
        file_path = None

        if is_local:
            file_path = self._resolve_module_path(module_name, current_file, level)

        return ImportInfo(
            module_name=module_name,
            imported_names=imported_names,
            is_local=is_local,
            file_path=file_path
        )

    def _is_local_module(self, module_name: str) -> bool:
        """Determine if module is local (not stdlib/third-party)."""
        base_module = module_name.split('.')[0]

        if base_module in self.STDLIB_MODULES:
            return False
        if base_module in self.THIRD_PARTY_MODULES:
            return False

        return True

    def _resolve_module_path(self, module_name: str, current_file: Path,
                            level: int) -> Optional[Path]:
        """Resolve module name to file path."""
        if level > 0:
            base_dir = current_file.parent
            for _ in range(level - 1):
                base_dir = base_dir.parent
        else:
            base_dir = self.root_dir

        module_path = base_dir / module_name.replace('.', '/')

        if (module_path.with_suffix('.py')).exists():
            return module_path.with_suffix('.py')

        if (module_path / '__init__.py').exists():
            return module_path / '__init__.py'

        return None

    def _file_to_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            rel_path = file_path.relative_to(self.root_dir)
            parts = list(rel_path.parts)
            if parts[-1] == '__init__.py':
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1].replace('.py', '')
            return '.'.join(parts)
        except ValueError:
            return file_path.stem

    def _topological_sort(self) -> list[DependencyNode]:
        """Sort dependencies in topological order."""
        visited = set()
        stack = []

        def dfs(module_name: str):
            if module_name in visited:
                return
            visited.add(module_name)

            node = self.resolved.get(module_name)
            if not node:
                return

            for imp in node.imports:
                if imp.is_local and imp.module_name in self.resolved:
                    dfs(imp.module_name)

            stack.append(node)

        for module_name in self.resolved:
            dfs(module_name)

        return stack
