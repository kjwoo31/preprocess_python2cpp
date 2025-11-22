"""Mapping Database for Python-to-C++ function mappings."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .validator import MappingValidator


@dataclass
class FunctionMapping:
    """
    Represents a mapping from Python function to C++ implementation.

    Attributes:
        python_lib: Python library name (e.g., 'cv2', 'numpy')
        python_func: Python function name (e.g., 'imread', 'reshape')
        cpp_lib: C++ library name (e.g., 'cv', 'Eigen')
        cpp_func: C++ function/method name
        cpp_header: Required C++ header files
        is_method: Whether this is a method call on an object
        arg_mapping: How to map arguments (optional)
        custom_template: Custom code template for complex mappings
        statements: List of C++ statements for N:M mapping
        inline_impl: Reference to implementation in implementations.yaml
        notes: Additional notes about the mapping
    """

    python_lib: str
    python_func: str
    cpp_lib: str
    cpp_func: str
    cpp_headers: list[str] = field(default_factory=list)
    is_method: bool = False
    arg_mapping: dict[str, str] | None = None
    custom_template: str | None = None
    statements: list[str] | None = None
    inline_impl: str | None = None
    notes: str = ""


@dataclass
class ConstantMapping:
    """
    Represents a mapping from Python constant to C++ constant.

    Attributes:
        python_lib: Python library name (e.g., 'cv2', 'numpy')
        python_const: Python constant name (e.g., 'IMREAD_GRAYSCALE')
        cpp_value: C++ constant value
        notes: Additional notes about the mapping
    """

    python_lib: str
    python_const: str
    cpp_value: str
    notes: str = ""


class MappingDatabase:
    """
    Database of Python-to-C++ function mappings.

    This class provides a searchable database of known mappings
    and can be extended with custom rules.
    """

    def __init__(self, auto_load_learned: bool = True, config_dir: str | None = None):
        """
        Initialize mapping database.

        Args:
            auto_load_learned: If True, automatically load learned_mappings.json
            config_dir: Path to config directory (default: project_root/config)
        """
        self.mappings: dict[str, FunctionMapping] = {}
        self.constants: dict[str, ConstantMapping] = {}
        self.implementations: dict[str, str] = {}  # impl_name -> code

        if config_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_dir = str(project_root / "config")

        self.config_dir = Path(config_dir)
        self._load_implementations()
        self._load_from_config_files()

    def _load_implementations(self) -> None:
        """Load C++ implementation snippets from config/implementations/."""
        impl_dir = self.config_dir / "implementations"
        if not impl_dir.exists():
            print(f"⚠️  implementations directory not found: {impl_dir}")
            return

        impl_files = list(impl_dir.glob("*.yaml"))
        if not impl_files:
            print(f"⚠️  No implementation files found in {impl_dir}")
            return

        for impl_file in impl_files:
            try:
                with open(impl_file) as f:
                    data = yaml.safe_load(f)

                if data:
                    self.implementations.update(data)
            except Exception as e:
                print(f"⚠️  Failed to load {impl_file.name}: {e}")

        if self.implementations:
            print(f"✓ Loaded {len(self.implementations)} C++ implementations")

    def _load_from_config_files(self) -> None:
        """Load mappings from config/mappings/ directory."""
        mappings_dir = self.config_dir / "mappings"
        if not mappings_dir.exists():
            print(f"⚠️  Mappings directory not found: {mappings_dir}")
            return

        config_files = list(mappings_dir.glob("*.yaml"))
        if not config_files:
            print(f"⚠️  No mapping files found in {mappings_dir}")
            return

        for config_path in config_files:
            self._load_config_file(config_path)

    def _load_config_file(self, config_path: Path) -> None:
        """Load mappings from a single YAML file."""
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return

            validation_errors = MappingValidator.validate_config_file(
                data, config_path.name
            )
            if validation_errors:
                print(f"⚠️  Validation errors in {config_path.name}:")
                for error in validation_errors:
                    print(f"   - {error}")
                return

            for func_data in data.get("functions", []):
                mapping = FunctionMapping(
                    python_lib=func_data["python_lib"],
                    python_func=func_data["python_func"],
                    cpp_lib=func_data["cpp_lib"],
                    cpp_func=func_data["cpp_func"],
                    cpp_headers=func_data.get("cpp_headers", []),
                    is_method=func_data.get("is_method", False),
                    arg_mapping=func_data.get("arg_mapping"),
                    custom_template=func_data.get("custom_template"),
                    statements=func_data.get("statements"),
                    inline_impl=func_data.get("inline_impl"),
                    notes=func_data.get("notes", ""),
                )
                self.add_mapping(mapping)

            for const_data in data.get("constants", []):
                constant = ConstantMapping(
                    python_lib=const_data["python_lib"],
                    python_const=const_data["python_const"],
                    cpp_value=const_data["cpp_value"],
                    notes=const_data.get("notes", ""),
                )
                self.add_constant(constant)

        except Exception as e:
            print(f"⚠️  Failed to load {config_path.name}: {e}")

    def add_mapping(self, mapping: FunctionMapping):
        """
        Add a new mapping to the database.

        Args:
            mapping: FunctionMapping to add
        """
        key = f"{mapping.python_lib}.{mapping.python_func}"
        self.mappings[key] = mapping

    def add_constant(self, constant: ConstantMapping):
        """
        Add a new constant mapping to the database.

        Args:
            constant: ConstantMapping to add
        """
        key = f"{constant.python_lib}.{constant.python_const}"
        self.constants[key] = constant

    def get_constant(
        self, python_lib: str, python_const: str
    ) -> ConstantMapping | None:
        """
        Get mapping for a Python constant.

        Args:
            python_lib: Python library name (e.g., 'cv2')
            python_const: Python constant name (e.g., 'IMREAD_GRAYSCALE')

        Returns:
            ConstantMapping if found, None otherwise
        """
        key = f"{python_lib}.{python_const}"
        return self.constants.get(key)

    def get_mapping(self, python_lib: str, python_func: str) -> FunctionMapping | None:
        """
        Get mapping for a Python function.

        Args:
            python_lib: Python library name
            python_func: Python function name

        Returns:
            FunctionMapping if found, None otherwise
        """
        key = f"{python_lib}.{python_func}"
        return self.mappings.get(key)

    def get_all_mappings(self) -> list[FunctionMapping]:
        """Get all mappings in the database"""
        return list(self.mappings.values())

    def get_required_headers(self, mappings: list[FunctionMapping]) -> list[str]:
        """
        Get all unique headers required for a list of mappings.

        Args:
            mappings: List of FunctionMapping objects

        Returns:
            List of unique header file includes
        """
        headers = set()
        for mapping in mappings:
            headers.update(mapping.cpp_headers)
        return sorted(headers)

    def export_to_json(self, filepath: str):
        """Export mappings to JSON file"""
        data = []
        for mapping in self.mappings.values():
            data.append(
                {
                    "python_lib": mapping.python_lib,
                    "python_func": mapping.python_func,
                    "cpp_lib": mapping.cpp_lib,
                    "cpp_func": mapping.cpp_func,
                    "cpp_headers": mapping.cpp_headers,
                    "is_method": mapping.is_method,
                    "notes": mapping.notes,
                }
            )

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filepath: str):
        """Import mappings from JSON file"""
        with open(filepath) as f:
            data = json.load(f)

        for item in data:
            mapping = FunctionMapping(
                python_lib=item["python_lib"],
                python_func=item["python_func"],
                cpp_lib=item["cpp_lib"],
                cpp_func=item["cpp_func"],
                cpp_headers=item["cpp_headers"],
                is_method=item.get("is_method", False),
                notes=item.get("notes", ""),
            )
            self.add_mapping(mapping)

    def save_learned_mapping(
        self, mapping: FunctionMapping, learned_file: str | None = None
    ) -> None:
        """
        Save a new learned mapping to the config/mappings/learned.yaml file.

        Args:
            mapping: The mapping to save
            learned_file: Path to learned mappings file (default: config/mappings/learned.yaml)
        """
        from pathlib import Path

        learned_path: Path
        if learned_file is None:
            learned_path = self.config_dir / "mappings" / "learned.yaml"
        else:
            learned_path = Path(learned_file)

        # Add to current database
        self.add_mapping(mapping)

        # Ensure directory exists
        if not learned_path.parent.exists():
            learned_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing learned mappings
        data: dict[str, list] = {"functions": [], "constants": []}
        if learned_path.exists():
            try:
                with open(learned_path) as f:
                    loaded_data = yaml.safe_load(f)
                    if loaded_data:
                        data = loaded_data
            except Exception as e:
                print(f"⚠️  Failed to load existing learned mappings: {e}")

        if "functions" not in data:
            data["functions"] = []

        # Check if this mapping already exists
        key = f"{mapping.python_lib}.{mapping.python_func}"
        existing_idx = -1
        for i, m in enumerate(data["functions"]):
            if f"{m['python_lib']}.{m['python_func']}" == key:
                existing_idx = i
                break

        mapping_dict = {
            "python_lib": mapping.python_lib,
            "python_func": mapping.python_func,
            "cpp_lib": mapping.cpp_lib,
            "cpp_func": mapping.cpp_func,
            "cpp_headers": mapping.cpp_headers,
            "is_method": mapping.is_method,
            "notes": mapping.notes + " [LLM-learned]",
        }

        # Add other optional fields if present
        if mapping.arg_mapping:
            mapping_dict["arg_mapping"] = mapping.arg_mapping
        if mapping.custom_template:
            mapping_dict["custom_template"] = mapping.custom_template
        if mapping.statements:
            mapping_dict["statements"] = mapping.statements
        if mapping.inline_impl:
            mapping_dict["inline_impl"] = mapping.inline_impl

        if existing_idx >= 0:
            # Update existing mapping
            data["functions"][existing_idx] = mapping_dict
        else:
            # Add new mapping
            data["functions"].append(mapping_dict)

        # Save back to file
        try:
            with open(learned_path, "w") as f:
                yaml.dump(data, f, sort_keys=False, indent=2)
            print(
                f"✓ Saved learned mapping to {learned_path.name}: {key} → {mapping.cpp_lib}::{mapping.cpp_func}"
            )
        except Exception as e:
            print(f"⚠️  Failed to save learned mapping: {e}")
