"""LLM-based Code Generator for unmapped operations."""

from typing import Dict, Any, Optional
import json
import os

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from core.intermediate.schema import IROperation, IRPipeline


class LLMCodeGenerator:
    """
    Generates C++ code using LLM for unmapped operations.

    This is used as a fallback when the mapping database doesn't
    have a direct Python-to-C++ mapping.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini"):
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

    def generate_cpp_for_operation(self, operation: IROperation,
                                   context: Dict[str, Any]) -> str:
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
        cpp_code = self._extract_code(response)

        return cpp_code

    def _build_prompt(self, operation: IROperation, context: Dict[str, Any]) -> str:
        """Build a prompt for the LLM using few-shot learning."""
        sections = [
            "You are an expert C++ programmer tasked with converting Python preprocessing operations to C++.",
            "",
            self._build_prompt_context(),
            self._build_prompt_operation(operation, context),
            self._build_prompt_task(operation),
            self._build_prompt_examples(operation),
            "## Your C++ Code",
            "Generate the C++ code below (only code, no explanations):",
            "",
            "```cpp"
        ]
        return '\n'.join(sections)

    def _build_prompt_context(self) -> str:
        """Build context section of the prompt."""
        return """## Context
You are converting a Python data preprocessing pipeline to C++. The available C++ libraries are:
- OpenCV (cv::) for image processing
- Eigen for linear algebra (NumPy equivalent)
- Standard C++17"""

    def _build_prompt_operation(self, operation: IROperation,
                                context: Dict[str, Any]) -> str:
        """Build operation section of the prompt."""
        return f"""## Current Operation
The operation to convert is represented in JSON (Intermediate Representation):

```json
{operation.to_dict()}
```

## Available Variables
{json.dumps(context.get('available_vars', {}), indent=2)}"""

    def _build_prompt_task(self, operation: IROperation) -> str:
        """Build task guidelines section of the prompt."""
        return f"""## Task
Generate ONLY the C++ code to implement this operation. Follow these guidelines:

1. Use modern C++17 features
2. Prefer OpenCV cv::Mat for image/array data
3. Use Eigen for mathematical operations if needed
4. Ensure memory safety (use smart pointers if needed)
5. Add brief inline comments explaining the logic
6. The output variable name must be: {operation.output}"""

    def _build_prompt_examples(self, operation: IROperation) -> str:
        """Build examples section of the prompt."""
        return f"""## Examples of Similar Conversions

Example 1 - Python to C++:
Python: `img = cv2.GaussianBlur(img, (5, 5), 0)`
C++:
```cpp
cv::Mat {operation.output};
cv::GaussianBlur(img, {operation.output}, cv::Size(5, 5), 0);
```

Example 2 - Python to C++:
Python: `result = np.mean(array, axis=0)`
C++:
```cpp
cv::Mat {operation.output};
cv::reduce(array, {operation.output}, 0, cv::REDUCE_AVG);
```"""

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM API.

        Args:
            prompt: Prompt to send

        Returns:
            LLM response text
        """
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Install with: pip install openai")

        try:
            # Using OpenAI Chat API (v1.0.0+)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert C++ programmer specializing in data preprocessing code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent code generation
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")

    def _extract_code(self, response: str) -> str:
        """
        Extract C++ code from LLM response.

        Args:
            response: Full LLM response

        Returns:
            Extracted C++ code
        """
        # Remove markdown code blocks if present
        lines = response.strip().split('\n')

        # Find code block boundaries
        in_code_block = False
        code_lines = []

        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue

            if in_code_block or not any(line.strip().startswith(x) for x in ['```', '#', '//']):
                code_lines.append(line)

        # If no code block markers, use entire response
        if not code_lines:
            code_lines = [line for line in lines if line.strip()]

        return '\n'.join(code_lines).strip()

    def validate_generated_code(self, code: str) -> bool:
        """
        Basic validation of generated C++ code.

        Args:
            code: Generated C++ code

        Returns:
            True if code appears valid
        """
        # Basic syntax checks
        if not code.strip():
            return False

        # Should contain C++ syntax
        cpp_keywords = ['cv::', 'Mat', 'std::', 'Eigen::', '=', ';']
        has_cpp = any(keyword in code for keyword in cpp_keywords)

        return has_cpp


class AnthropicLLMGenerator(LLMCodeGenerator):
    """
    LLM generator using Anthropic Claude API.

    Alternative to OpenAI for code generation.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
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
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")

    def _call_llm(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return message.content[0].text

        except Exception as e:
            raise RuntimeError(f"Claude API call failed: {e}")


class VertexAILLMGenerator(LLMCodeGenerator):
    """
    LLM generator using Anthropic Claude via Google Cloud Vertex AI.

    Uses Google Cloud's Vertex AI to access Claude models.
    Requires: ANTHROPIC_VERTEX_PROJECT_ID, CLOUD_ML_REGION env vars.
    """

    def __init__(self, project_id: Optional[str] = None,
                 region: Optional[str] = None,
                 model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Vertex AI Claude generator.

        Args:
            project_id: GCP project ID (or set ANTHROPIC_VERTEX_PROJECT_ID env var)
            region: GCP region (or set CLOUD_ML_REGION env var)
            model: Claude model to use via Vertex AI
        """
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
            raise ValueError(
                "CLAUDE_CODE_USE_VERTEX=1 must be set to use Vertex AI"
            )

        try:
            import anthropic
            self.client = anthropic.AnthropicVertex(
                project_id=self.project_id,
                region=self.region
            )
            print(f"✓ Using Vertex AI (Project: {self.project_id}, Region: {self.region})")
        except ImportError:
            raise ImportError(
                "anthropic[vertex] package not installed. "
                "Install with: pip install 'anthropic[vertex]'"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Vertex AI client: {e}")

    def _call_llm(self, prompt: str) -> str:
        """Call Anthropic Claude via Vertex AI"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return message.content[0].text

        except Exception as e:
            raise RuntimeError(f"Vertex AI API call failed: {e}")
