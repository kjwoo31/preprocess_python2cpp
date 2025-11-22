"""Code Mapper for IR to C++ operation mapping."""

from core.intermediate.schema import IROperation, IRPipeline, OperationType

from .database import FunctionMapping, MappingDatabase


class CodeMapper:
    """
    Maps IR operations to C++ equivalents using the mapping database.

    This class bridges the IR and C++ code generation phases.
    """

    def __init__(self, mapping_db: MappingDatabase | None = None):
        """
        Initialize the mapper.

        Args:
            mapping_db: Optional custom mapping database.
                       If None, uses default database.
        """
        self.db = mapping_db or MappingDatabase()

    def map_operation(self, operation: IROperation) -> FunctionMapping | None:
        """
        Map an IR operation to C++ function.

        Args:
            operation: IR operation to map

        Returns:
            FunctionMapping if mapping exists, None otherwise
        """
        if operation.op_type == OperationType.FUNCTION_CALL:
            return self.db.get_mapping(operation.source_lib, operation.function)

        elif operation.op_type == OperationType.METHOD_CALL:
            # Try to get mapping for method
            # First try with full library path
            mapping = self.db.get_mapping(operation.source_lib, operation.function)
            if mapping:
                return mapping

            # Try common patterns
            # Check if it's a numpy array method
            if (
                operation.source_lib == "numpy"
                or "ndarray" in operation.source_object.lower()
            ):
                mapping = self.db.get_mapping("numpy.ndarray", operation.function)
                if mapping:
                    return mapping

        elif operation.op_type == OperationType.ARITHMETIC:
            # Arithmetic operations are handled directly, no mapping needed
            return None

        return None

    def get_required_headers(self, pipeline: IRPipeline) -> list[str]:
        """
        Get all C++ headers required for a pipeline.

        Args:
            pipeline: IR pipeline

        Returns:
            List of required header includes
        """
        headers = set()

        # Always include standard headers
        headers.add("<iostream>")
        headers.add("<string>")
        headers.add("<vector>")

        # Add headers based on operations
        for op in pipeline.operations:
            mapping = self.map_operation(op)
            if mapping:
                headers.update(mapping.cpp_headers)

        return sorted(headers)

    def get_unmapped_operations(self, pipeline: IRPipeline) -> list[IROperation]:
        """
        Find operations that don't have mappings.

        Args:
            pipeline: IR pipeline

        Returns:
            List of operations without mappings
        """
        unmapped = []

        for op in pipeline.operations:
            if op.op_type in [OperationType.FUNCTION_CALL, OperationType.METHOD_CALL]:
                mapping = self.map_operation(op)
                if not mapping:
                    unmapped.append(op)

        return unmapped

    def get_required_libraries(self, pipeline: IRPipeline) -> set[str]:
        """
        Get set of C++ libraries required for pipeline.

        Args:
            pipeline: IR pipeline

        Returns:
            Set of library names (e.g., {'OpenCV', 'Eigen', 'FFTW'})
        """
        libraries = set()

        for op in pipeline.operations:
            mapping = self.map_operation(op)
            if mapping:
                libraries.add(mapping.cpp_lib)

        return libraries

    def suggest_cmake_packages(self, pipeline: IRPipeline) -> list[str]:
        """
        Suggest CMake find_package() calls for the pipeline.

        Args:
            pipeline: IR pipeline

        Returns:
            List of CMake package names
        """
        libs = self.get_required_libraries(pipeline)

        cmake_map = {
            "cv": "OpenCV",
            "Eigen": "Eigen3",
            "fftw": "FFTW3",
            "sndfile": "SndFile",
        }

        packages = []
        for lib in libs:
            if lib in cmake_map:
                packages.append(cmake_map[lib])

        return packages
