"""C++ Code Generator for IR-to-C++ project conversion."""

from pathlib import Path

from core.intermediate.schema import IROperation, IRPipeline
from core.mapping.core import CodeMapper

from .template import TemplateEngine

try:
    from .llm_provider import (
        AnthropicLLMGenerator,
        LLMCodeGenerator,
        VertexAILLMGenerator,
    )

    HAS_LLM = True
except ImportError:
    HAS_LLM = False


class CodeGenerator:
    """
    Generates C++ code from IR pipeline.

    This class orchestrates the code generation process:
    1. Maps IR operations to C++ equivalents
    2. Determines required libraries and headers
    3. Renders C++ code using templates
    4. Generates build files (CMakeLists.txt)
    5. (Optional) Uses LLM for unmapped operations
    """

    def __init__(
        self,
        output_dir: str = "output",
        use_llm: bool = False,
        llm_provider: str = "openai",
        llm_api_key: str | None = None,
        save_learned: bool = True,
    ):
        """Initialize code generator."""
        self.output_dir = Path(output_dir)
        self.mapper = CodeMapper()
        self.template_engine = TemplateEngine()
        self.save_learned = save_learned
        self.use_llm = use_llm
        self.llm_generator: LLMCodeGenerator | None = None

        if use_llm:
            self._initialize_llm_generator(llm_provider, llm_api_key, save_learned)

    def _initialize_llm_generator(
        self, llm_provider: str, llm_api_key: str | None, save_learned: bool
    ) -> None:
        """Initialize LLM generator based on provider."""
        if not HAS_LLM:
            raise ImportError(
                "LLM support requires additional packages. "
                "Install with: pip install openai anthropic"
            )

        if llm_provider == "openai":
            self.llm_generator = LLMCodeGenerator(api_key=llm_api_key)
            print(f"✓ LLM integration enabled (OpenAI {self.llm_generator.model})")
        elif llm_provider == "anthropic":
            self.llm_generator = AnthropicLLMGenerator(api_key=llm_api_key)
            print("✓ LLM integration enabled (Anthropic Claude)")
        elif llm_provider == "vertex":
            self.llm_generator = VertexAILLMGenerator()
        else:
            raise ValueError(
                f"Unknown LLM provider: {llm_provider}. Choose: openai, anthropic, or vertex"
            )

        if save_learned:
            print("✓ Auto-save learned mappings enabled")

    def generate(self, pipeline: IRPipeline, project_name: str | None = None) -> str:
        """
        Generate complete C++ project from IR pipeline.

        Args:
            pipeline: IR pipeline to generate code for
            project_name: Optional project name (defaults to pipeline name)

        Returns:
            Path to generated project directory
        """
        if project_name is None:
            project_name = pipeline.name

        project_dir = self.output_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        self._generate_cpp_file(pipeline, project_dir)
        self._generate_cmake_file(pipeline, project_dir, project_name)
        self._copy_stb_headers(project_dir)
        self._generate_validator_header(project_dir)
        self._generate_readme(pipeline, project_dir, project_name)

        return str(project_dir)

    def generate_multi_function(
        self,
        pipelines: list[IRPipeline],
        project_name: str,
        main_functions: list[str],
        main_block_operations: list,
    ) -> str:
        """
        Generate complete C++ project from multiple IR pipelines (one file).

        Args:
            pipelines: List of IR pipelines to generate code for
            project_name: Project name (usually the source file name)
            main_functions: List of function names called in Python's __main__ block
            main_block_operations: IR operations from __main__ block

        Returns:
            Path to generated project directory
        """
        project_dir = self.output_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        self._generate_multi_cpp_file(
            pipelines, project_dir, project_name, main_functions, main_block_operations
        )
        self._generate_multi_cmake_file(pipelines, project_dir, project_name)
        self._copy_stb_headers(project_dir)
        self._generate_validator_header(project_dir)
        self._generate_multi_readme(pipelines, project_dir, project_name)

        return str(project_dir)

    def _build_operation_mappings(self, pipeline: IRPipeline) -> dict:
        """Build operation mappings for template."""
        mappings = {}
        for op in pipeline.operations:
            mapping = self.mapper.map_operation(op)
            if mapping:
                mappings[op.id] = mapping
        return mappings

    def _process_unmapped_operations(self, pipeline: IRPipeline) -> dict[str, str]:
        """
        Process unmapped operations using LLM.

        Args:
            pipeline: IR pipeline

        Returns:
            Dictionary mapping operation ID to generated C++ code
        """
        if not self.use_llm or not self.llm_generator:
            return {}

        unmapped_ops = self.mapper.get_unmapped_operations(pipeline)
        if not unmapped_ops:
            return {}

        print(
            f"\n🤖 Using LLM to generate code for {len(unmapped_ops)} unmapped operation(s)..."
        )
        return self._generate_code_for_operations(unmapped_ops)

    def _generate_code_for_operations(self, unmapped_ops: list) -> dict[str, str]:
        """Generate C++ code for list of unmapped operations."""
        generated_code = {}

        for op in unmapped_ops:
            print(f"  Generating: {op.source_lib}.{op.function} ({op.id})")

            context = {"available_vars": {}, "operation": op.to_dict()}

            code = self._try_generate_operation_code(op, context)
            if code:
                generated_code[op.id] = code

        return generated_code

    def _try_generate_operation_code(self, op, context: dict) -> str | None:
        """Try to generate code for single operation."""
        try:
            if not self.llm_generator:
                return None

            cpp_code = self.llm_generator.generate_cpp_for_operation(op, context)

            if not cpp_code:
                return None

            if self.llm_generator.validate_generated_code(cpp_code):
                print("    ✓ Generated successfully")
                if self.save_learned:
                    self._save_llm_as_mapping(op, cpp_code)
                return cpp_code
            else:
                print("    ⚠️ Generated code failed validation")
                return None

        except Exception as e:
            print(f"    ✗ LLM generation failed: {e}")
            return None

    def _save_llm_as_mapping(self, operation: IROperation, cpp_code: str):
        """
        Extract mapping information from LLM-generated code and save it.

        Args:
            operation: The IR operation that was converted
            cpp_code: The C++ code generated by LLM
        """

        try:
            is_method = operation.op_type.value == "method_call"
            cpp_lib, cpp_func = self._extract_cpp_lib_func(
                cpp_code, operation, is_method
            )
            cpp_headers = self._extract_cpp_headers(cpp_code)

            if cpp_func:
                mapping = self._create_function_mapping(
                    operation, cpp_lib, cpp_func, cpp_headers, is_method
                )
                self.mapper.db.save_learned_mapping(mapping)

        except Exception as e:
            print(f"    ℹ️  Could not extract mapping (will use inline code): {e}")

    def _extract_cpp_lib_func(
        self, cpp_code: str, operation: IROperation, is_method: bool
    ) -> tuple[str | None, str | None]:
        """Extract C++ library and function name from generated code."""
        import re

        namespace_pattern = r"(\w+)::(\w+)"
        matches = re.findall(namespace_pattern, cpp_code)

        if matches:
            return matches[0]
        elif is_method:
            method_pattern = r"\.(\w+)\("
            method_match = re.search(method_pattern, cpp_code)
            if method_match:
                return operation.source_lib, method_match.group(1)

        return None, None

    def _extract_cpp_headers(self, cpp_code: str) -> list[str]:
        """Extract header includes from C++ code."""
        import re

        header_pattern = r'#include\s+[<"](.+?)[>"]'
        headers = re.findall(header_pattern, cpp_code)

        if headers:
            return [f"<{h}>" if not h.startswith("<") else h for h in headers]
        return ["<iostream>"]

    def _create_function_mapping(
        self,
        operation: IROperation,
        cpp_lib: str | None,
        cpp_func: str,
        cpp_headers: list[str],
        is_method: bool,
    ):
        """Create FunctionMapping from extracted information."""
        from core.mapping.database import FunctionMapping

        return FunctionMapping(
            python_lib=operation.source_lib or "unknown",
            python_func=operation.function,
            cpp_lib=cpp_lib or "custom",
            cpp_func=cpp_func,
            cpp_headers=cpp_headers,
            is_method=is_method,
            notes=f"LLM-generated from {operation.id}",
        )

    def _generate_cpp_file(self, pipeline: IRPipeline, project_dir: Path):
        """Generate main C++ source file"""
        # Process unmapped operations with LLM if enabled
        llm_generated_code = {}
        if self.use_llm:
            llm_generated_code = self._process_unmapped_operations(pipeline)

        headers = self.mapper.get_required_headers(pipeline)

        libs = self.mapper.get_required_libraries(pipeline)
        has_eigen = "Eigen" in libs
        has_opencv = "cv" in libs

        operation_mappings = {}
        for op in pipeline.operations:
            mapping = self.mapper.map_operation(op)
            if mapping:
                operation_mappings[op.id] = mapping

        context = {
            "pipeline": pipeline,
            "headers": headers,
            "has_eigen": has_eigen,
            "has_opencv": has_opencv,
            "llm_generated_code": llm_generated_code,
            "operation_mappings": operation_mappings,
            "implementations": self.mapper.db.implementations,
        }

        cpp_code = self.template_engine.render_cpp_code("cpp/base.cpp.j2", context)

        output_file = project_dir / f"{pipeline.name}.cpp"
        with open(output_file, "w") as f:
            f.write(cpp_code)

        print(f"Generated C++ file: {output_file}")

    def _generate_multi_cpp_file(
        self,
        pipelines: list[IRPipeline],
        project_dir: Path,
        project_name: str,
        main_functions: list[str],
        main_block_operations: list,
    ):
        """Generate C++ source file with multiple functions"""
        all_headers = set()
        all_libs = set()
        all_llm_code = {}
        all_operation_mappings = {}
        all_implementations = set()

        pipeline_data = []

        for pipeline in pipelines:
            if self.use_llm:
                llm_code = self._process_unmapped_operations(pipeline)
                all_llm_code.update(llm_code)

            headers = self.mapper.get_required_headers(pipeline)
            all_headers.update(headers)

            libs = self.mapper.get_required_libraries(pipeline)
            all_libs.update(libs)

            operation_mappings = {}
            for op in pipeline.operations:
                mapping = self.mapper.map_operation(op)
                if mapping:
                    operation_mappings[op.id] = mapping
                    if mapping.inline_impl:
                        all_implementations.add(mapping.inline_impl)

            all_operation_mappings.update(operation_mappings)

            pipeline_data.append(
                {
                    "pipeline": pipeline,
                    "operation_mappings": operation_mappings,
                }
            )

        # Build operation mappings for main block
        main_block_mappings = {}
        for op in main_block_operations:
            mapping = self.mapper.map_operation(op)
            if mapping:
                main_block_mappings[op.id] = mapping
                if mapping.inline_impl:
                    all_implementations.add(mapping.inline_impl)

        context = {
            "project_name": project_name,
            "pipelines": pipeline_data,
            "main_functions": main_functions,
            "main_block_operations": main_block_operations,
            "main_block_mappings": main_block_mappings,
            "headers": sorted(all_headers),
            "has_eigen": "Eigen" in all_libs,
            "has_opencv": "cv" in all_libs,
            "llm_generated_code": all_llm_code,
            "all_operation_mappings": all_operation_mappings,
            "implementations": self.mapper.db.implementations,
            "all_implementations": all_implementations,
        }

        cpp_code = self.template_engine.render_cpp_code("cpp/multi.cpp.j2", context)

        output_file = project_dir / f"{project_name}.cpp"
        with open(output_file, "w") as f:
            f.write(cpp_code)

        print(f"Generated C++ file: {output_file}")

    def _generate_cmake_file(
        self, pipeline: IRPipeline, project_dir: Path, project_name: str
    ):
        """Generate CMakeLists.txt"""
        cmake_packages = self.mapper.suggest_cmake_packages(pipeline)

        libs = self.mapper.get_required_libraries(pipeline)

        context = {
            "project_name": project_name,
            "source_file": f"{pipeline.name}.cpp",
            "cmake_packages": cmake_packages,
            "has_eigen": "Eigen" in libs,
            "has_opencv": "cv" in libs,
            "has_fftw": "fftw" in libs,
            "has_sndfile": "sndfile" in libs,
        }

        cmake_content = self.template_engine.render_cmake(
            "cmake/cmakelists.txt.j2", context
        )

        output_file = project_dir / "CMakeLists.txt"
        with open(output_file, "w") as f:
            f.write(cmake_content)

        print(f"Generated CMake file: {output_file}")

    def _generate_multi_cmake_file(
        self, pipelines: list[IRPipeline], project_dir: Path, project_name: str
    ):
        """Generate CMakeLists.txt for multi-function project"""
        all_packages = set()
        all_libs = set()

        for pipeline in pipelines:
            cmake_packages = self.mapper.suggest_cmake_packages(pipeline)
            all_packages.update(cmake_packages)

            libs = self.mapper.get_required_libraries(pipeline)
            all_libs.update(libs)

        context = {
            "project_name": project_name,
            "source_file": f"{project_name}.cpp",
            "cmake_packages": sorted(all_packages),
            "has_eigen": "Eigen" in all_libs,
            "has_opencv": "cv" in all_libs,
            "has_fftw": "fftw" in all_libs,
            "has_sndfile": "sndfile" in all_libs,
        }

        cmake_content = self.template_engine.render_cmake(
            "cmake/cmakelists.txt.j2", context
        )

        output_file = project_dir / "CMakeLists.txt"
        with open(output_file, "w") as f:
            f.write(cmake_content)

        print(f"Generated CMake file: {output_file}")

    def _generate_image_header(self, project_dir: Path):
        """Generate image.h header file (supports JPEG/PNG via stb)"""
        image_content = self.template_engine.render_image_header()

        output_file = project_dir / "image.h"
        with open(output_file, "w") as f:
            f.write(image_content)

        print(f"Generated image header: {output_file}")

    def _copy_stb_headers(self, project_dir: Path):
        """Copy stb_image headers to project directory"""
        import shutil

        src_root = Path(__file__).parent.parent.parent
        stb_src_dir = src_root / "templates" / "headers"

        stb_files = ["stb_image.h", "stb_image_write.h"]
        for stb_file in stb_files:
            src_path = stb_src_dir / stb_file
            dst_path = project_dir / stb_file
            if src_path.exists():
                shutil.copy(src_path, dst_path)
                print(f"Copied {stb_file}: {dst_path}")

    def _generate_validator_header(self, project_dir: Path):
        """Generate validator.h header file"""
        validator_content = self.template_engine.render_validator_header()

        output_file = project_dir / "validator.h"
        with open(output_file, "w") as f:
            f.write(validator_content)

        print(f"Generated validator header: {output_file}")

    def _generate_readme(
        self, pipeline: IRPipeline, project_dir: Path, project_name: str
    ):
        """Generate README.md with build instructions."""
        unmapped = self.mapper.get_unmapped_operations(pipeline)
        packages = self.mapper.suggest_cmake_packages(pipeline)

        sections = self._build_readme_sections(
            pipeline, project_name, packages, unmapped
        )
        readme_content = "\n".join(sections)

        output_file = project_dir / "README.md"
        output_file.write_text(readme_content)

        print(f"Generated README: {output_file}")

    def _generate_multi_readme(
        self, pipelines: list[IRPipeline], project_dir: Path, project_name: str
    ):
        """Generate README.md for multi-function project."""
        all_unmapped = []
        all_packages = set()

        for pipeline in pipelines:
            unmapped = self.mapper.get_unmapped_operations(pipeline)
            all_unmapped.extend(unmapped)
            packages = self.mapper.suggest_cmake_packages(pipeline)
            all_packages.update(packages)

        function_names = [p.name for p in pipelines]
        sections = self._build_multi_readme_sections(
            pipelines, project_name, function_names, sorted(all_packages), all_unmapped
        )
        readme_content = "\n".join(sections)

        output_file = project_dir / "README.md"
        output_file.write_text(readme_content)

        print(f"Generated README: {output_file}")

    def _build_readme_sections(
        self, pipeline: IRPipeline, project_name: str, packages: list, unmapped: list
    ) -> list[str]:
        """Build all README sections."""
        return [
            self._build_readme_header(pipeline, project_name),
            self._build_requirements_section(packages),
            self._build_build_instructions(project_name),
            self._build_pipeline_info(pipeline),
            self._build_unmapped_section(unmapped),
            self._build_validation_section(),
            self._build_notes_section(),
        ]

    def _build_readme_header(self, pipeline: IRPipeline, project_name: str) -> str:
        """Build README header section."""
        return f"""# {project_name}

Auto-generated C++ implementation of Python preprocessing pipeline: `{pipeline.name}`

## Generated Files

- `{pipeline.name}.cpp` - Main C++ implementation
- `CMakeLists.txt` - CMake build configuration
- `validator.h` - Numerical validation utilities
"""

    def _build_requirements_section(self, packages: list[str]) -> str:
        """Build requirements section."""
        package_map = {
            "OpenCV": "- OpenCV 4.x",
            "Eigen3": "- Eigen 3.4+",
            "FFTW3": "- FFTW3",
            "SndFile": "- libsndfile",
        }

        requirements = [package_map[pkg] for pkg in packages if pkg in package_map]
        requirements_text = "\n".join(requirements)

        return f"""## Build Instructions

### Requirements

{requirements_text}
"""

    def _build_build_instructions(self, project_name: str) -> str:
        """Build compilation and execution instructions."""
        return f"""### Build Steps

```bash
mkdir build
cd build
cmake ..
make
```

### Run

```bash
# Run all functions (like Python's if __name__ == "__main__")
./{project_name} <input_file>
```
"""

    def _build_pipeline_info(self, pipeline: IRPipeline) -> str:
        """Build pipeline information section."""
        libraries = ", ".join(pipeline.metadata.get("libraries", []))
        num_operations = len(pipeline.operations)

        return f"""## Pipeline Information

**Source Libraries:** {libraries}

**Operations:** {num_operations}
"""

    def _build_unmapped_section(self, unmapped: list) -> str:
        """Build unmapped operations warning section."""
        if not unmapped:
            return ""

        unmapped_items = "\n".join(
            f"- `{op.source_lib}.{op.function}` (Operation ID: {op.id})"
            for op in unmapped
        )

        return f"""
## ⚠️ Unmapped Operations

The following operations do not have automatic mappings and may require manual implementation:

{unmapped_items}
"""

    def _build_validation_section(self) -> str:
        """Build validation instructions section."""
        return """
## Validation

To validate the generated C++ code against Python output:

1. Run the Python version and save outputs to `.npy` files
2. Run the C++ version
3. Use the validation utilities in `validator.h` to compare results
"""

    def _build_notes_section(self) -> str:
        """Build notes section."""
        return """
## Notes

This code was auto-generated by the Python-to-C++ Porting Agent.
Manual review and testing is recommended before production use.
"""

    def _build_multi_readme_sections(
        self,
        pipelines: list[IRPipeline],
        project_name: str,
        function_names: list[str],
        packages: list[str],
        unmapped: list,
    ) -> list[str]:
        """Build README sections for multi-function project."""
        return [
            self._build_multi_readme_header(pipelines, project_name, function_names),
            self._build_requirements_section(packages),
            self._build_build_instructions(project_name),
            self._build_multi_pipeline_info(pipelines),
            self._build_unmapped_section(unmapped),
            self._build_validation_section(),
            self._build_notes_section(),
        ]

    def _build_multi_readme_header(
        self, pipelines: list[IRPipeline], project_name: str, function_names: list[str]
    ) -> str:
        """Build README header for multi-function project."""
        functions_list = "\n".join(f"- `{name}`" for name in function_names)
        return f"""# {project_name}

Auto-generated C++ implementation of Python preprocessing functions.

## Functions

{functions_list}

## Generated Files

- `{project_name}.cpp` - C++ implementation with all functions
- `CMakeLists.txt` - CMake build configuration
- `validator.h` - Numerical validation utilities
"""

    def _build_multi_pipeline_info(self, pipelines: list[IRPipeline]) -> str:
        """Build pipeline information for multi-function project."""
        all_libs = set()
        total_ops = 0

        for pipeline in pipelines:
            all_libs.update(pipeline.metadata.get("libraries", []))
            total_ops += len(pipeline.operations)

        libraries = ", ".join(sorted(all_libs))

        return f"""## Pipeline Information

**Source Libraries:** {libraries}

**Total Operations:** {total_ops}

**Functions:** {len(pipelines)}
"""

    def generate_report(self, pipeline: IRPipeline) -> str:
        """
        Generate a report about the conversion.

        Args:
            pipeline: IR pipeline

        Returns:
            Report as string
        """
        unmapped = self.mapper.get_unmapped_operations(pipeline)
        libs = self.mapper.get_required_libraries(pipeline)

        report = f"""
=== Code Generation Report ===

Pipeline: {pipeline.name}
Operations: {len(pipeline.operations)}
Inputs: {len(pipeline.inputs)}
Outputs: {len(pipeline.outputs)}

Required C++ Libraries:
{chr(10).join(f"  - {lib}" for lib in sorted(libs))}

Unmapped Operations: {len(unmapped)}
"""

        if unmapped:
            report += "\nOperations requiring manual implementation:\n"
            for op in unmapped:
                report += f"  - {op.source_lib}.{op.function} ({op.id})\n"

        return report
