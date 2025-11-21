"""C++ Code Generator for IR-to-C++ project conversion."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.intermediate.schema import IRPipeline, IROperation, OperationType
from core.mapping.core import CodeMapper
from core.mapping.database import MappingDatabase
from .template import TemplateEngine

try:
    from .llm_provider import LLMCodeGenerator, AnthropicLLMGenerator, VertexAILLMGenerator
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

    def __init__(self, output_dir: str = "output", use_llm: bool = False,
                 llm_provider: str = "openai", llm_api_key: Optional[str] = None,
                 save_learned: bool = True):
        """
        Initialize code generator.

        Args:
            output_dir: Directory for generated code
            use_llm: Enable LLM for unmapped operations (requires API key)
            llm_provider: LLM provider ('openai', 'anthropic', or 'vertex')
            llm_api_key: API key for LLM service (or use environment variable)
            save_learned: If True, save LLM-generated mappings for future reuse
        """
        self.output_dir = Path(output_dir)
        self.mapper = CodeMapper()
        self.template_engine = TemplateEngine()
        self.save_learned = save_learned

        # LLM integration
        self.use_llm = use_llm
        self.llm_generator = None

        if use_llm:
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
                # Message already printed in VertexAILLMGenerator.__init__
            else:
                raise ValueError(f"Unknown LLM provider: {llm_provider}. Choose: openai, anthropic, or vertex")

            if save_learned:
                print("✓ Auto-save learned mappings enabled")

    def generate(self, pipeline: IRPipeline, project_name: Optional[str] = None) -> str:
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

        # Create project directory
        project_dir = self.output_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # Generate files
        self._generate_cpp_file(pipeline, project_dir)
        self._generate_cmake_file(pipeline, project_dir, project_name)
        self._generate_image_header(project_dir)
        self._copy_stb_headers(project_dir)
        self._generate_validator_header(project_dir)
        self._generate_readme(pipeline, project_dir, project_name)

        return str(project_dir)

    def _process_unmapped_operations(self, pipeline: IRPipeline) -> Dict[str, str]:
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

        print(f"\n🤖 Using LLM to generate code for {len(unmapped_ops)} unmapped operation(s)...")

        generated_code = {}

        for op in unmapped_ops:
            print(f"  Generating: {op.source_lib}.{op.function} ({op.id})")

            # Build context for LLM
            context = {
                'available_vars': {},  # TODO: Track variable types
                'operation': op.to_dict()
            }

            try:
                cpp_code = self.llm_generator.generate_cpp_for_operation(op, context)

                # Validate generated code
                if self.llm_generator.validate_generated_code(cpp_code):
                    generated_code[op.id] = cpp_code
                    print(f"    ✓ Generated successfully")

                    # Save as learned mapping if enabled
                    if self.save_learned:
                        self._save_llm_as_mapping(op, cpp_code)
                else:
                    print(f"    ⚠️ Generated code failed validation")

            except Exception as e:
                print(f"    ✗ LLM generation failed: {e}")

        return generated_code

    def _save_llm_as_mapping(self, operation: IROperation, cpp_code: str):
        """
        Extract mapping information from LLM-generated code and save it.

        Args:
            operation: The IR operation that was converted
            cpp_code: The C++ code generated by LLM
        """
        from core.mapping.database import FunctionMapping

        try:
            is_method = operation.op_type.value == 'method_call'
            cpp_lib, cpp_func = self._extract_cpp_lib_func(cpp_code, operation, is_method)
            cpp_headers = self._extract_cpp_headers(cpp_code)

            if cpp_func:
                mapping = self._create_function_mapping(
                    operation, cpp_lib, cpp_func, cpp_headers, is_method
                )
                self.mapper.db.save_learned_mapping(mapping)

        except Exception as e:
            print(f"    ℹ️  Could not extract mapping (will use inline code): {e}")

    def _extract_cpp_lib_func(self, cpp_code: str, operation: IROperation,
                              is_method: bool) -> tuple[Optional[str], Optional[str]]:
        """Extract C++ library and function name from generated code."""
        import re

        namespace_pattern = r'(\w+)::(\w+)'
        matches = re.findall(namespace_pattern, cpp_code)

        if matches:
            return matches[0]
        elif is_method:
            method_pattern = r'\.(\w+)\('
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
            return [f'<{h}>' if not h.startswith('<') else h for h in headers]
        return ['<iostream>']

    def _create_function_mapping(self, operation: IROperation, cpp_lib: Optional[str],
                                 cpp_func: str, cpp_headers: list[str],
                                 is_method: bool):
        """Create FunctionMapping from extracted information."""
        from core.mapping.database import FunctionMapping

        return FunctionMapping(
            python_lib=operation.source_lib or 'unknown',
            python_func=operation.function,
            cpp_lib=cpp_lib or 'custom',
            cpp_func=cpp_func,
            cpp_headers=cpp_headers,
            is_method=is_method,
            notes=f'LLM-generated from {operation.id}'
        )

    def _generate_cpp_file(self, pipeline: IRPipeline, project_dir: Path):
        """Generate main C++ source file"""
        # Process unmapped operations with LLM if enabled
        llm_generated_code = {}
        if self.use_llm:
            llm_generated_code = self._process_unmapped_operations(pipeline)

        # Get required headers
        headers = self.mapper.get_required_headers(pipeline)

        # Determine which libraries are used
        libs = self.mapper.get_required_libraries(pipeline)
        has_eigen = 'Eigen' in libs
        has_opencv = 'cv' in libs

        # Build operation mappings for template
        operation_mappings = {}
        for op in pipeline.operations:
            mapping = self.mapper.map_operation(op)
            if mapping:
                operation_mappings[op.id] = mapping

        # Prepare template context
        context = {
            'pipeline': pipeline,
            'headers': headers,
            'has_eigen': has_eigen,
            'has_opencv': has_opencv,
            'llm_generated_code': llm_generated_code,
            'operation_mappings': operation_mappings,
        }

        # Render template
        cpp_code = self.template_engine.render_cpp_code('cpp/base.cpp.j2', context)

        # Write to file
        output_file = project_dir / f"{pipeline.name}.cpp"
        with open(output_file, 'w') as f:
            f.write(cpp_code)

        print(f"Generated C++ file: {output_file}")

    def _generate_cmake_file(self, pipeline: IRPipeline, project_dir: Path, project_name: str):
        """Generate CMakeLists.txt"""
        # Get required CMake packages
        cmake_packages = self.mapper.suggest_cmake_packages(pipeline)

        # Determine which libraries are used
        libs = self.mapper.get_required_libraries(pipeline)

        context = {
            'project_name': project_name,
            'source_file': f"{pipeline.name}.cpp",
            'cmake_packages': cmake_packages,
            'has_eigen': 'Eigen' in libs,
            'has_opencv': 'cv' in libs,
            'has_fftw': 'fftw' in libs,
            'has_sndfile': 'sndfile' in libs,
        }

        cmake_content = self.template_engine.render_cmake('cmake/cmakelists.txt.j2', context)

        output_file = project_dir / "CMakeLists.txt"
        with open(output_file, 'w') as f:
            f.write(cmake_content)

        print(f"Generated CMake file: {output_file}")

    def _generate_image_header(self, project_dir: Path):
        """Generate image.h header file (supports JPEG/PNG via stb)"""
        image_content = self.template_engine.render_image_header()

        output_file = project_dir / "image.h"
        with open(output_file, 'w') as f:
            f.write(image_content)

        print(f"Generated image header: {output_file}")

    def _copy_stb_headers(self, project_dir: Path):
        """Copy stb_image headers to project directory"""
        import shutil

        src_root = Path(__file__).parent.parent.parent
        stb_src_dir = src_root / 'templates' / 'headers'

        stb_files = ['stb_image.h', 'stb_image_write.h']
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
        with open(output_file, 'w') as f:
            f.write(validator_content)

        print(f"Generated validator header: {output_file}")

    def _generate_readme(self, pipeline: IRPipeline, project_dir: Path, project_name: str):
        """Generate README.md with build instructions."""
        unmapped = self.mapper.get_unmapped_operations(pipeline)
        packages = self.mapper.suggest_cmake_packages(pipeline)

        sections = [
            self._build_readme_header(pipeline, project_name),
            self._build_requirements_section(packages),
            self._build_build_instructions(project_name),
            self._build_pipeline_info(pipeline),
            self._build_unmapped_section(unmapped),
            self._build_validation_section(),
            self._build_notes_section()
        ]

        readme_content = '\n'.join(sections)

        output_file = project_dir / "README.md"
        with open(output_file, 'w') as f:
            f.write(readme_content)

        print(f"Generated README: {output_file}")

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
            'OpenCV': '- OpenCV 4.x',
            'Eigen3': '- Eigen 3.4+',
            'FFTW3': '- FFTW3',
            'SndFile': '- libsndfile'
        }

        requirements = [package_map[pkg] for pkg in packages if pkg in package_map]
        requirements_text = '\n'.join(requirements)

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
./{project_name} <input_file>
```
"""

    def _build_pipeline_info(self, pipeline: IRPipeline) -> str:
        """Build pipeline information section."""
        libraries = ", ".join(pipeline.metadata.get('libraries', []))
        num_operations = len(pipeline.operations)

        return f"""## Pipeline Information

**Source Libraries:** {libraries}

**Operations:** {num_operations}
"""

    def _build_unmapped_section(self, unmapped: list) -> str:
        """Build unmapped operations warning section."""
        if not unmapped:
            return ""

        unmapped_items = '\n'.join(
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
