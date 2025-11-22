"""Schema validation for mapping configuration files."""

from typing import Any


class MappingValidator:
    """Validates mapping configuration files against schema."""

    REQUIRED_FUNCTION_FIELDS = ["python_lib", "python_func", "cpp_lib", "cpp_func"]

    REQUIRED_CONSTANT_FIELDS = ["python_lib", "python_const", "cpp_value"]

    OPTIONAL_FUNCTION_FIELDS = [
        "cpp_headers",
        "is_method",
        "arg_mapping",
        "custom_template",
        "statements",
        "notes",
    ]

    OPTIONAL_CONSTANT_FIELDS = ["notes"]

    @staticmethod
    def validate_function_mapping(func_data: dict[str, Any]) -> list[str]:
        """
        Validate a single function mapping entry.

        Args:
            func_data: Dictionary containing function mapping data

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for field in MappingValidator.REQUIRED_FUNCTION_FIELDS:
            if field not in func_data:
                errors.append(f"Missing required field: {field}")

        if "cpp_headers" in func_data and not isinstance(
            func_data["cpp_headers"], list
        ):
            errors.append("cpp_headers must be a list")

        if "is_method" in func_data and not isinstance(func_data["is_method"], bool):
            errors.append("is_method must be a boolean")

        if (
            "statements" in func_data
            and func_data["statements"] is not None
            and not isinstance(func_data["statements"], list)
        ):
            errors.append("statements must be a list or null")

        return errors

    @staticmethod
    def validate_constant_mapping(const_data: dict[str, Any]) -> list[str]:
        """
        Validate a single constant mapping entry.

        Args:
            const_data: Dictionary containing constant mapping data

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for field in MappingValidator.REQUIRED_CONSTANT_FIELDS:
            if field not in const_data:
                errors.append(f"Missing required field: {field}")

        return errors

    @staticmethod
    def validate_config_file(data: dict[str, Any], filename: str) -> list[str]:
        """
        Validate entire configuration file.

        Args:
            data: Parsed YAML data
            filename: Name of the config file (for error messages)

        Returns:
            List of all validation errors
        """
        all_errors = []

        if "functions" in data:
            if not isinstance(data["functions"], list):
                all_errors.append(f"{filename}: 'functions' must be a list")
            else:
                for idx, func_data in enumerate(data["functions"]):
                    errors = MappingValidator.validate_function_mapping(func_data)
                    for error in errors:
                        all_errors.append(
                            f"{filename}: Function[{idx}] ({func_data.get('python_func', '?')}): {error}"
                        )

        if "constants" in data:
            if not isinstance(data["constants"], list):
                all_errors.append(f"{filename}: 'constants' must be a list")
            else:
                for idx, const_data in enumerate(data["constants"]):
                    errors = MappingValidator.validate_constant_mapping(const_data)
                    for error in errors:
                        all_errors.append(
                            f"{filename}: Constant[{idx}] ({const_data.get('python_const', '?')}): {error}"
                        )

        return all_errors
