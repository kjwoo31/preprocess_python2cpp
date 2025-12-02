"""LLM-based Code Generator for unmapped operations."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from core.intermediate.schema import IROperation

DEFAULT_MAX_ITERATIONS = 3
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS_OPENAI = 800
DEFAULT_MAX_TOKENS_ANTHROPIC = 1024
COMPILE_TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = (
    "You are an expert C++ programmer specializing in converting Python "
    "data preprocessing code to Modern C++17. You strictly follow RAII, "
    "const correctness, and never use raw pointers for ownership. "
    "You write clean, self-documenting code that adheres to best practices."
)

CPP_INDICATORS = ("cv::", "Mat", "std::", "Eigen::", "=", ";")
FORBIDDEN_PATTERNS = ("new ", "delete ", " malloc(", " free(", "using namespace std")

CPP_TEST_TEMPLATE = """
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

int main() {{
    cv::Mat image = cv::Mat::zeros(224, 224, CV_8UC3);
    cv::Mat result;
    {code}
    return 0;
}}
"""


def test_cpp_compilation(cpp_code: str, include_dirs: list[str] | None = None) -> bool:
    """
    Test if C++ code compiles successfully.

    Args:
        cpp_code: C++ code snippet to test.
        include_dirs: Optional list of include directories.

    Returns:
        True if compilation succeeds.
    """
    full_code = CPP_TEST_TEMPLATE.format(code=cpp_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "test.cpp"
        exe_file = Path(tmpdir) / "test"
        src_file.write_text(full_code)

        cmd = _build_compile_command(str(src_file), str(exe_file), include_dirs)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_SECONDS
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


def _build_compile_command(
    src_path: str, exe_path: str, include_dirs: list[str] | None
) -> list[str]:
    """Build g++ compile command with flags."""
    cmd = ["g++", "-std=c++17", src_path, "-o", exe_path]
    cmd.extend(["-I/usr/include/opencv4", "-lopencv_core", "-lopencv_imgproc"])

    if include_dirs:
        for inc_dir in include_dirs:
            cmd.extend(["-I", inc_dir])

    return cmd


class LLMCodeGenerator:
    """Generates C++ code using LLM for unmapped operations."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Initialize LLM code generator.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var).
            model: Model to use (gpt-4, gpt-4o-mini, etc.).
            max_iterations: Max attempts for compilable solution.
            temperature: LLM temperature (0.0 = deterministic, no retry).
        """
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        if HAS_OPENAI:
            self.client = openai.OpenAI(api_key=self.api_key)

    def generate_cpp_for_operation(
        self, operation: IROperation, context: dict[str, Any]
    ) -> str | None:
        """
        Generate C++ code for a single operation using LLM.

        Args:
            operation: IR operation to convert.
            context: Additional context (available variables, types, etc.).

        Returns:
            Generated C++ code snippet or None.
        """
        prompt = self._build_prompt(operation, context)
        response = self._call_llm(prompt)

        if response:
            return self._extract_code(response)
        return None

    def generate_with_compile_retry(
        self,
        operation: IROperation,
        context: dict[str, Any],
        compile_check: bool = True,
    ) -> tuple[str | None, bool]:
        """
        Generate C++ code with compile verification and retry.

        Args:
            operation: IR operation to convert.
            context: Additional context.
            compile_check: Whether to verify compilation.

        Returns:
            Tuple of (generated code, compilation success).
        """
        is_deterministic = self.temperature == 0.0
        effective_max = 1 if is_deterministic else self.max_iterations

        if is_deterministic:
            print("      (deterministic: temperature=0, no retry)")

        for attempt in range(1, effective_max + 1):
            if effective_max > 1:
                print(f"      Attempt {attempt}/{effective_max}...")

            result = self._try_single_generation(operation, context, compile_check)
            if result:
                return result, True

        return None, False

    def _try_single_generation(
        self, operation: IROperation, context: dict[str, Any], compile_check: bool
    ) -> str | None:
        """Attempt single code generation and validation."""
        cpp_code = self.generate_cpp_for_operation(operation, context)

        if not cpp_code:
            print("      ✗ No code generated")
            return None

        if not self.validate_generated_code(cpp_code):
            print("      ✗ Code validation failed")
            return None

        if not compile_check:
            return cpp_code

        if test_cpp_compilation(cpp_code):
            print("      ✓ Compilation successful")
            return cpp_code

        print("      ✗ Compilation failed")
        return None

    def _build_prompt(self, operation: IROperation, context: dict[str, Any]) -> str:
        """Build prompt for LLM using few-shot learning."""
        sections = [
            "You are an expert C++ programmer specializing in Python-to-C++ conversion.",
            "",
            self._build_prompt_context(),
            "",
            self._build_prompt_operation(operation, context),
            "",
            self._build_prompt_task(operation),
            "",
            self._build_prompt_examples(operation),
            "",
            self._build_prompt_output_requirements(),
        ]
        return "\n".join(sections)

    def _build_prompt_context(self) -> str:
        """Build context section."""
        return """## Context
You are converting a Python data preprocessing pipeline to Modern C++17.

### Available C++ Libraries
- **OpenCV (cv::)**: Image processing, computer vision operations
- **Eigen**: Linear algebra, NumPy-equivalent matrix operations
- **STL (C++17)**: Standard library containers and algorithms
- **stb_image.h**: Lightweight image I/O (header-only, already included)

### Type Mappings
- `numpy.ndarray` → `cv::Mat` (for image/array data)
- `numpy.ndarray` → `Eigen::MatrixXd` (for mathematical operations)
- `list[T]` → `std::vector<T>`
- `tuple[T1, T2]` → `std::pair<T1, T2>` or `std::tuple<T1, T2>`
- `str` → `std::string`
- `int`, `float`, `bool` → native C++ types"""

    def _build_prompt_operation(
        self, operation: IROperation, context: dict[str, Any]
    ) -> str:
        """Build operation section."""
        return f"""## Current Operation
The operation to convert is represented in JSON (Intermediate Representation):

```json
{operation.to_dict()}
```

## Available Variables
{json.dumps(context.get("available_vars", {}), indent=2)}"""

    def _build_prompt_task(self, operation: IROperation) -> str:
        """Build task guidelines section."""
        return f"""## Task
Generate ONLY the C++ code to implement this operation.

### Required Guidelines (MUST follow)
1. **Output variable name**: MUST be `{operation.output}`
2. **Modern C++17**: Use auto, structured bindings, std::optional when appropriate
3. **RAII principle**: All resources must use RAII (no manual new/delete)
4. **No raw pointers**: Use `std::unique_ptr`, `std::shared_ptr`, or references
5. **Const correctness**: Mark immutable variables and parameters as `const`
6. **Type safety**: Prefer `static_cast<>` over C-style casts

### Forbidden Practices
- Raw pointers for ownership: `T* ptr = new T()`
- C-style casts: `(int)value`
- Manual memory management: `delete ptr`
- Global variables or static state
- `using namespace std;` in headers"""

    def _build_prompt_examples(self, operation: IROperation) -> str:
        """Build examples section."""
        output_var = operation.output
        examples = [
            self._example_resize(output_var),
            self._example_color_convert(output_var),
            self._example_normalize(output_var),
            self._example_reduce(output_var),
        ]
        return "## Examples of Correct Conversions\n\n" + "\n\n".join(examples)

    def _example_resize(self, output_var: str) -> str:
        """Image resize example."""
        return f"""### Image Resize
**Python:** `img = cv2.resize(image, (224, 224))`
**C++:**
```cpp
cv::Mat {output_var};
cv::resize(image, {output_var}, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
```"""

    def _example_color_convert(self, output_var: str) -> str:
        """Color conversion example."""
        return f"""### Color Conversion
**Python:** `rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)`
**C++:**
```cpp
cv::Mat {output_var};
cv::cvtColor(bgr_img, {output_var}, cv::COLOR_BGR2RGB);
```"""

    def _example_normalize(self, output_var: str) -> str:
        """Normalization example."""
        return f"""### Type Conversion with Normalization
**Python:** `normalized = img.astype(np.float32) / 255.0`
**C++:**
```cpp
cv::Mat {output_var};
image.convertTo({output_var}, CV_32F, 1.0 / 255.0);
```"""

    def _example_reduce(self, output_var: str) -> str:
        """Array reduction example."""
        return f"""### Array Reduction
**Python:** `mean_val = np.mean(array, axis=0)`
**C++:**
```cpp
cv::Mat {output_var};
cv::reduce(array, {output_var}, 0, cv::REDUCE_AVG);
```"""

    def _build_prompt_output_requirements(self) -> str:
        """Build output requirements section."""
        return """## Output Requirements
Generate ONLY the C++ code that implements the operation above.
- Do NOT include explanations or markdown code blocks
- Include necessary variable declarations
- Ensure the code compiles

Your C++ code:"""

    def _call_llm(self, prompt: str) -> str | None:
        """
        Call the OpenAI API.

        Args:
            prompt: Prompt to send.

        Returns:
            LLM response text or None.

        Raises:
            ImportError: If openai package not installed.
            RuntimeError: If API call fails.
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=DEFAULT_MAX_TOKENS_OPENAI,
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def _extract_code(self, response: str) -> str:
        """Extract C++ code from LLM response, removing markdown blocks."""
        lines = response.strip().split("\n")
        in_code_block = False
        code_lines = []

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block or not line.strip().startswith(("#", "//")):
                code_lines.append(line)

        if not code_lines:
            code_lines = [line for line in lines if line.strip()]

        return "\n".join(code_lines).strip()

    def validate_generated_code(self, code: str) -> bool:
        """
        Validate generated C++ code for quality and safety.

        Args:
            code: Generated C++ code.

        Returns:
            True if code appears valid and safe.
        """
        if not code.strip():
            return False

        has_cpp_syntax = any(indicator in code for indicator in CPP_INDICATORS)
        if not has_cpp_syntax:
            return False

        for pattern in FORBIDDEN_PATTERNS:
            if pattern in code:
                print(f"    ⚠️  Forbidden pattern found: {pattern!r}")
                return False

        return True


class AnthropicLLMGenerator(LLMCodeGenerator):
    """LLM generator using Anthropic Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Initialize Anthropic Claude generator.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
            model: Claude model to use.
            max_iterations: Max attempts for compilable solution.
            temperature: LLM temperature (0.0 = deterministic, no retry).
        """
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter"
            )

        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _call_llm(self, prompt: str) -> str | None:
        """Call Anthropic Claude API."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS_ANTHROPIC,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}") from e


class FallbackLLMGenerator(LLMCodeGenerator):
    """
    LLM generator with OpenAI → Anthropic fallback.

    Tries OpenAI first. If generation fails, falls back to Anthropic Claude.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
        anthropic_model: str = "claude-sonnet-4-20250514",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Initialize fallback generator.

        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var).
            anthropic_api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
            openai_model: OpenAI model to use.
            anthropic_model: Anthropic model to use.
            max_iterations: Max attempts per provider.
            temperature: LLM temperature (0.0 = deterministic, no retry).
        """
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.openai_generator = self._init_openai(openai_api_key, openai_model)
        self.anthropic_generator = self._init_anthropic(anthropic_api_key, anthropic_model)

        if not self.openai_generator and not self.anthropic_generator:
            raise ValueError(
                "At least one API key required. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
            )

        self.model = (
            self.openai_generator.model
            if self.openai_generator
            else self.anthropic_generator.model
        )

    def _init_openai(
        self, api_key: str | None, model: str
    ) -> LLMCodeGenerator | None:
        """Initialize OpenAI generator if available."""
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        if openai_key and HAS_OPENAI:
            return LLMCodeGenerator(
                api_key=openai_key,
                model=model,
                max_iterations=self.max_iterations,
                temperature=self.temperature,
            )
        return None

    def _init_anthropic(
        self, api_key: str | None, model: str
    ) -> AnthropicLLMGenerator | None:
        """Initialize Anthropic generator if available."""
        anthropic_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and HAS_ANTHROPIC:
            return AnthropicLLMGenerator(
                api_key=anthropic_key,
                model=model,
                max_iterations=self.max_iterations,
                temperature=self.temperature,
            )
        return None

    def generate_cpp_for_operation(
        self, operation: IROperation, context: dict[str, Any]
    ) -> str | None:
        """Generate using primary provider."""
        if self.openai_generator:
            return self.openai_generator.generate_cpp_for_operation(operation, context)
        if self.anthropic_generator:
            return self.anthropic_generator.generate_cpp_for_operation(operation, context)
        return None

    def generate_with_compile_retry(
        self,
        operation: IROperation,
        context: dict[str, Any],
        compile_check: bool = True,
    ) -> tuple[str | None, bool]:
        """
        Generate C++ code with OpenAI → Anthropic fallback.

        Args:
            operation: IR operation to convert.
            context: Additional context.
            compile_check: Whether to verify compilation.

        Returns:
            Tuple of (generated code, compilation success).
        """
        if self.openai_generator:
            code, success = self._try_provider(
                self.openai_generator, "OpenAI", operation, context, compile_check
            )
            if success:
                return code, True

        if self.anthropic_generator:
            code, success = self._try_provider(
                self.anthropic_generator, "Anthropic", operation, context, compile_check
            )
            if success:
                return code, True

        return None, False

    def _try_provider(
        self,
        generator: LLMCodeGenerator,
        provider_name: str,
        operation: IROperation,
        context: dict[str, Any],
        compile_check: bool,
    ) -> tuple[str | None, bool]:
        """Try generation with a specific provider."""
        icon = "🔷" if provider_name == "OpenAI" else "🔶"
        print(f"    {icon} Trying {provider_name} ({generator.model})...")

        code, success = generator.generate_with_compile_retry(
            operation, context, compile_check
        )

        if not success:
            effective_attempts = 1 if self.temperature == 0.0 else self.max_iterations
            print(f"    ✗ {provider_name} failed after {effective_attempts} attempt(s)")

        return code, success

    def validate_generated_code(self, code: str) -> bool:
        """Validate using primary provider."""
        if self.openai_generator:
            return self.openai_generator.validate_generated_code(code)
        if self.anthropic_generator:
            return self.anthropic_generator.validate_generated_code(code)
        return False
