"""LLM-based Code Generator for unmapped operations."""

import json
import os
from typing import Any

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from core.intermediate.schema import IROperation


class LLMCodeGenerator:
    """
    Generates C++ code using LLM for unmapped operations.

    This is used as a fallback when the mapping database doesn't
    have a direct Python-to-C++ mapping.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-5-mini"):
        """
        Initialize LLM code generator.

        Args:
            api_key: API key for the LLM service (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-3.5-turbo, claude-3-opus, etc.)
        """
        self.model = model
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
            operation: IR operation to convert
            context: Additional context (available variables, types, etc.)

        Returns:
            Generated C++ code snippet
        """
        prompt = self._build_prompt(operation, context)

        response = self._call_llm(prompt)

        # Extract C++ code from response
        if response:
            cpp_code = self._extract_code(response)
            return cpp_code
        return None

    def _build_prompt(self, operation: IROperation, context: dict[str, Any]) -> str:
        """Build a prompt for the LLM using few-shot learning."""
        sections = [
            "You are an expert C++ programmer specializing in Python-to-C++ conversion for data preprocessing pipelines.",
            "",
            self._build_prompt_context(),
            "",
            self._build_prompt_operation(operation, context),
            "",
            self._build_prompt_task(operation),
            "",
            self._build_prompt_examples(operation),
            "",
            "## Output Requirements",
            "Generate ONLY the C++ code that implements the operation above.",
            "- Do NOT include explanations, comments about what you're doing, or markdown",
            "- Do NOT wrap in ```cpp blocks (I will do that)",
            "- Include necessary variable declarations",
            "- Ensure the code compiles and follows all guidelines above",
            "",
            "Your C++ code:",
        ]
        return "\n".join(sections)

    def _build_prompt_context(self) -> str:
        """Build context section of the prompt."""
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
        """Build operation section of the prompt."""
        return f"""## Current Operation
The operation to convert is represented in JSON (Intermediate Representation):

```json
{operation.to_dict()}
```

## Available Variables
{json.dumps(context.get("available_vars", {}), indent=2)}"""

    def _build_prompt_task(self, operation: IROperation) -> str:
        """Build task guidelines section of the prompt."""
        return f"""## Task
Generate ONLY the C++ code to implement this operation.

### Required Guidelines (MUST follow)
1. **Output variable name**: MUST be `{operation.output}`
2. **Modern C++17**: Use auto, structured bindings, std::optional when appropriate
3. **RAII principle**: All resources must use RAII (no manual new/delete)
4. **No raw pointers**: Use `std::unique_ptr`, `std::shared_ptr`, or references
5. **Const correctness**: Mark immutable variables and parameters as `const`
6. **Type safety**: Prefer `static_cast<>` over C-style casts

### Data Structure Preferences
- Image/array data → `cv::Mat` (OpenCV)
- Mathematical operations → `Eigen::MatrixXd` or `cv::Mat`
- Collections → `std::vector<T>`, `std::array<T, N>`
- Strings → `std::string`

### Code Quality
- Self-documenting: Use clear, descriptive variable names
- Minimal comments: Only explain "why", not "what"
- Error handling: Check for edge cases (null pointers, empty arrays, invalid dimensions)
- No magic numbers: Use named constants

### Forbidden Practices
- ❌ Raw pointers for ownership: `T* ptr = new T()`
- ❌ C-style casts: `(int)value`
- ❌ Manual memory management: `delete ptr`
- ❌ Global variables or static state
- ❌ `using namespace std;` in headers"""

    def _build_prompt_examples(self, operation: IROperation) -> str:
        """Build examples section of the prompt."""
        output_var = operation.output
        return f"""## Examples of Correct Conversions

### Example 1: Image Processing (OpenCV)
**Python:**
```python
img = cv2.resize(image, (224, 224))
```

**C++ (Correct):**
```cpp
cv::Mat {output_var};
cv::resize(image, {output_var}, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
```

### Example 2: Color Conversion
**Python:**
```python
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
```

**C++ (Correct):**
```cpp
cv::Mat {output_var};
cv::cvtColor(bgr_img, {output_var}, cv::COLOR_BGR2RGB);
```

### Example 3: Array Operations with Type Conversion
**Python:**
```python
normalized = img.astype(np.float32) / 255.0
```

**C++ (Correct):**
```cpp
cv::Mat {output_var};
image.convertTo({output_var}, CV_32F, 1.0 / 255.0);
```

### Example 4: Conditional Operations
**Python:**
```python
result = x if x > 0 else 0
```

**C++ (Correct):**
```cpp
const auto {output_var} = std::max(x, 0);
```

### Example 5: Array Reduction
**Python:**
```python
mean_val = np.mean(array, axis=0)
```

**C++ (Correct):**
```cpp
cv::Mat {output_var};
cv::reduce(array, {output_var}, 0, cv::REDUCE_AVG);
```

### Common Patterns
- **Shape operations**: `cv::Mat::reshape()`, `cv::transpose()`
- **Element-wise ops**: `cv::add()`, `cv::subtract()`, `cv::multiply()`, `cv::divide()`
- **Aggregations**: `cv::reduce()` with `cv::REDUCE_SUM`, `cv::REDUCE_AVG`, `cv::REDUCE_MAX`
- **Type conversions**: `mat.convertTo(dest, type, scale, offset)`"""

    def _call_llm(self, prompt: str) -> str | None:
        """
        Call the LLM API.

        Args:
            prompt: Prompt to send

        Returns:
            LLM response text
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert C++ programmer specializing in converting Python "
                            "data preprocessing code to Modern C++17. You strictly follow RAII, "
                            "const correctness, and never use raw pointers for ownership. "
                            "You write clean, self-documenting code that adheres to best practices."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=800,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e

    def _extract_code(self, response: str) -> str:
        """
        Extract C++ code from LLM response.

        Args:
            response: Full LLM response

        Returns:
            Extracted C++ code
        """
        # Remove markdown code blocks if present
        lines = response.strip().split("\n")

        # Find code block boundaries
        in_code_block = False
        code_lines = []

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block or not any(
                line.strip().startswith(x) for x in ["```", "#", "//"]
            ):
                code_lines.append(line)

        # If no code block markers, use entire response
        if not code_lines:
            code_lines = [line for line in lines if line.strip()]

        return "\n".join(code_lines).strip()

    def validate_generated_code(self, code: str) -> bool:
        """
        Validate generated C++ code for quality and safety.

        Args:
            code: Generated C++ code

        Returns:
            True if code appears valid and safe
        """
        if not code.strip():
            return False

        cpp_indicators = ["cv::", "Mat", "std::", "Eigen::", "=", ";"]
        has_cpp = any(indicator in code for indicator in cpp_indicators)

        if not has_cpp:
            return False

        forbidden_patterns = [
            "new ",
            "delete ",
            " malloc(",
            " free(",
            "using namespace std",
        ]
        has_forbidden = any(pattern in code for pattern in forbidden_patterns)

        if has_forbidden:
            print(f"    ⚠️  Generated code contains forbidden patterns: {code[:100]}...")
            return False

        return True


class AnthropicLLMGenerator(LLMCodeGenerator):
    """
    LLM generator using Anthropic Claude API.

    Alternative to OpenAI for code generation.
    """

    def __init__(
        self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize Anthropic Claude generator.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter"
            )

        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            ) from e

    def _call_llm(self, prompt: str) -> str | None:
        """Call Anthropic Claude API."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                system=(
                    "You are an expert C++ programmer specializing in converting Python "
                    "data preprocessing code to Modern C++17. You strictly follow RAII, "
                    "const correctness, and never use raw pointers for ownership. "
                    "You write clean, self-documenting code that adheres to best practices."
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            return message.content[0].text

        except Exception as e:
            raise RuntimeError(f"Claude API call failed: {e}") from e


class VertexAILLMGenerator(LLMCodeGenerator):
    """
    LLM generator using Anthropic Claude via Google Cloud Vertex AI.

    Uses Google Cloud's Vertex AI to access Claude models.
    Requires: ANTHROPIC_VERTEX_PROJECT_ID, CLOUD_ML_REGION env vars.
    """

    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize Vertex AI Claude generator.

        Args:
            project_id: GCP project ID (or set ANTHROPIC_VERTEX_PROJECT_ID env var)
            region: GCP region (or set CLOUD_ML_REGION env var)
            model: Claude model to use via Vertex AI
        """
        from anthropic import NotGiven

        self.model = model
        self.project_id = project_id or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
        self.region = region or os.getenv("CLOUD_ML_REGION", "us-east5")

        if not self.project_id:
            raise ValueError(
                "GCP project ID required. Set ANTHROPIC_VERTEX_PROJECT_ID environment variable "
                "or pass project_id parameter"
            )

        # Check if Vertex AI mode is enabled
        use_vertex = os.getenv("CLAUDE_CODE_USE_VERTEX", "0") == "1"
        if not use_vertex:
            raise ValueError("CLAUDE_CODE_USE_VERTEX=1 must be set to use Vertex AI")

        try:
            import anthropic

            self.client = anthropic.AnthropicVertex(
                project_id=self.project_id, region=self.region or NotGiven()
            )
            print(
                f"✓ Using Vertex AI (Project: {self.project_id}, Region: {self.region})"
            )
        except ImportError as e:
            raise ImportError(
                "anthropic[vertex] package not installed. "
                "Install with: pip install 'anthropic[vertex]'"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Vertex AI client: {e}") from e

    def _call_llm(self, prompt: str) -> str | None:
        """Call Anthropic Claude via Vertex AI."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                system=(
                    "You are an expert C++ programmer specializing in converting Python "
                    "data preprocessing code to Modern C++17. You strictly follow RAII, "
                    "const correctness, and never use raw pointers for ownership. "
                    "You write clean, self-documenting code that adheres to best practices."
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            return message.content[0].text

        except Exception as e:
            raise RuntimeError(f"Vertex AI API call failed: {e}") from e
