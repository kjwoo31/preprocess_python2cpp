"""IR Schema Definitions for language-neutral pipeline representation."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class OperationType(Enum):
    """Types of operations in the IR"""

    FUNCTION_CALL = "function_call"
    METHOD_CALL = "method_call"
    ARITHMETIC = "arithmetic"
    ASSIGNMENT = "assignment"
    CONDITIONAL = "conditional"
    LOOP = "loop"


@dataclass
class TypeHint:
    """
    Represents type information for variables in the IR.

    Attributes:
        base_type: The base type (e.g., 'numpy.ndarray', 'str', 'int')
        dtype: Data type for arrays (e.g., 'uint8', 'float32')
        shape: Shape information for arrays (e.g., (H, W, 3))
        is_const: Whether this is a constant value
    """

    base_type: str
    dtype: str | None = None
    shape: tuple | str | None = None
    is_const: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __str__(self) -> str:
        """String representation for debugging."""
        parts = [self.base_type]
        if self.dtype:
            parts.append(f"dtype={self.dtype}")
        if self.shape:
            parts.append(f"shape={self.shape}")
        return f"TypeHint({', '.join(parts)})"


@dataclass
class IRInput:
    """
    Represents an input parameter to the pipeline.

    Attributes:
        name: Variable name
        type_hint: Type information
        value: Default value (if any)
    """

    name: str
    type_hint: TypeHint
    value: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"name": self.name, "type": str(self.type_hint)}
        if self.value is not None:
            result["value"] = self.value
        return result


@dataclass
class IROperation:
    """
    Represents a single operation in the pipeline.

    Attributes:
        id: Unique operation identifier (e.g., 'op_1')
        op_type: Type of operation
        output: Variable name for the result
        output_type_hint: Type hint for the output
        source_lib: Source Python library (e.g., 'cv2', 'numpy')
        function: Function or method name
        source_object: For method calls, the object being called on
        args: List of arguments (variable names or literals)
        kwargs: Dictionary of keyword arguments
        operator: For arithmetic operations, the operator (e.g., 'add', 'divide')
        operands: For arithmetic operations, the operands
        condition: For conditionals, the condition expression
        true_branch: For conditionals, operations in if block
        false_branch: For conditionals, operations in else block
        loop_var: For loops, the iteration variable
        iterable: For loops, the iterable expression
        loop_body: For loops, operations in loop body
    """

    id: str
    op_type: OperationType
    output: str
    output_type_hint: TypeHint
    source_lib: str | None = None
    function: str | None = None
    source_object: str | None = None
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    operator: str | None = None
    operands: list[Any] = field(default_factory=list)
    condition: str | None = None
    true_branch: list["IROperation"] = field(default_factory=list)
    false_branch: list["IROperation"] = field(default_factory=list)
    loop_var: str | None = None
    iterable: str | None = None
    loop_body: list["IROperation"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "type": self.op_type.value,
            "output": self.output,
            "output_type_hint": str(self.output_type_hint),
        }

        self._add_optional_fields(result)
        self._add_list_fields(result)

        return result

    def _add_optional_fields(self, result: dict[str, Any]) -> None:
        """Add optional fields if present."""
        optional_fields = ["source_lib", "function", "source_object", "operator"]
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value

    def _add_list_fields(self, result: dict[str, Any]) -> None:
        """Add list fields if non-empty."""
        if self.args:
            result["args"] = self.args
        if self.kwargs:
            result["kwargs"] = self.kwargs
        if self.operands:
            result["operands"] = self.operands
        if self.condition:
            result["condition"] = self.condition
        if self.true_branch:
            result["true_branch"] = [op.to_dict() for op in self.true_branch]
        if self.false_branch:
            result["false_branch"] = [op.to_dict() for op in self.false_branch]
        if self.loop_var:
            result["loop_var"] = self.loop_var
        if self.iterable:
            result["iterable"] = self.iterable
        if self.loop_body:
            result["loop_body"] = [op.to_dict() for op in self.loop_body]


@dataclass
class IROutput:
    """
    Represents an output of the pipeline.

    Attributes:
        name: Variable name to return
        type_hint: Type information
    """

    name: str
    type_hint: TypeHint

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"name": self.name, "type": str(self.type_hint)}


@dataclass
class IRPipeline:
    """
    Represents a complete preprocessing pipeline.

    Attributes:
        name: Pipeline name (used for function naming)
        inputs: List of input parameters
        operations: List of operations in execution order
        outputs: List of output variables
        metadata: Additional metadata (e.g., source file, line numbers)
    """

    name: str
    inputs: list[IRInput]
    operations: list[IROperation]
    outputs: list[IROutput]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pipeline_name": self.name,
            "inputs": [inp.to_dict() for inp in self.inputs],
            "operations": [op.to_dict() for op in self.operations],
            "outputs": [out.to_dict() for out in self.outputs],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IRPipeline":
        """Create IRPipeline from dictionary."""
        raise NotImplementedError("Deserialization not yet implemented")

    def validate(self) -> list[str]:
        """Validate the IR for consistency."""
        errors = []
        defined_vars = {inp.name for inp in self.inputs}

        for op in self.operations:
            defined_vars.add(op.output)

        for output in self.outputs:
            if output.name not in defined_vars:
                errors.append(f"Output '{output.name}' not defined in operations")

        return errors
