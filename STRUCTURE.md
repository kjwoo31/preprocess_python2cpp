# Project Architecture

**Pattern**: Pipeline (Analysis → IR → Mapping → Generation)
**Version**: v1.3
**Last Updated**: 2025-12-02

---

## 📁 Directory Structure

```
preprocess_python2cpp/
├── README.md                     # User guide and quick start
├── STRUCTURE.md                  # Architecture documentation (this file)
├── CLAUDE.md                     # Coding standards and guidelines
├── tasks.md                      # Development roadmap
│
├── config/                       # Configuration files
│   ├── mappings/                 # Python → C++ function mappings
│   │   ├── opencv.yaml           # cv2.* → img::* mappings
│   │   ├── numpy.yaml            # numpy.* mappings
│   │   ├── librosa.yaml          # librosa.* mappings
│   │   └── pil.yaml              # PIL.* mappings
│   │
│   └── implementations/          # C++ inline implementations
│       └── img.yaml              # img::* functions (header-free)
│
├── src/
│   ├── cli/
│   │   └── main.py               # Entry point (< 50 lines per function)
│   │
│   ├── core/
│   │   ├── analysis/             # Python code analysis
│   │   │   ├── parser.py         # AST parsing
│   │   │   ├── inferencer.py    # Type inference
│   │   │   ├── separator.py     # Pipeline separation (Pre/Inf/Post)
│   │   │   ├── dependencies.py  # Dependency resolution
│   │   │   └── tracer.py        # Execution tracing
│   │   │
│   │   ├── intermediate/         # IR (Intermediate Representation)
│   │   │   ├── schema.py         # IR data structures
│   │   │   └── builder.py        # AST → IR conversion
│   │   │
│   │   ├── mapping/              # Python-to-C++ mappings
│   │   │   ├── database.py       # Loads from config/mappings/
│   │   │   ├── core.py           # Mapping logic
│   │   │   └── validator.py     # YAML validation
│   │   │
│   │   ├── generation/           # C++ code generation
│   │   │   ├── generator.py     # Main code generator
│   │   │   ├── template.py       # Jinja2 template engine
│   │   │   ├── filters.py        # Custom Jinja2 filters
│   │   │   └── llm_provider.py  # LLM integration (optional)
│   │   │
│   │   └── validation/           # Auto-validation
│   │       ├── executor.py       # Build/run executor
│   │       └── comparator.py    # Result comparison
│   │
│   └── templates/                # Jinja2 templates
│       ├── cpp/
│       │   ├── base.cpp.j2       # Single function template
│       │   └── multi.cpp.j2      # Multi-function template
│       ├── cmake/
│       │   └── cmakelists.txt.j2
│       └── headers/
│           ├── stb_image.h       # Image I/O (header-only)
│           ├── stb_image_write.h
│           └── validator.h.j2    # Validation utilities
│
├── examples/                     # Example Python code
│   └── vision/                   # Image processing examples
│       ├── image_preprocessing.py
│       └── mnist_inference.py
│
├── tests/                        # Test suite
│   ├── unit/
│   └── benchmarks/
│
└── .build/                       # Generated output (gitignored)
    └── output/                   # Generated C++ projects
```

---

## 🏗️ Core Components

### 1️⃣ CLI Layer (`src/cli/main.py`)

**Purpose**: Entry point and workflow orchestration

**Key Functions**:
- `create_argument_parser()`: Configure CLI arguments
- `parse_python_file()`: Load and parse Python source
- `_process_conversions()`: Orchestrate conversion pipeline
- `validate_multi_functions()`: Run validation

**Design**: All functions < 50 lines, follows SRP

---

### 2️⃣ Analysis Layer (`src/core/analysis/`)

#### Parser (`parser.py`)
- **Pattern**: Visitor Pattern for AST traversal
- **Extracts**: Functions, imports, assignments, calls
- **Output**: `FunctionInfo`, `ImportInfo` dataclasses

#### Type Inferencer (`inferencer.py`)
- **Strategy**: Annotations → Literals → Library signatures
- **Coverage**: 85% auto-inferred
- **Fallback**: `auto` type for unknown cases

#### Pipeline Separator (`separator.py`)
- **Marker**: `# @inference` comment
- **Splits**: Pre/Inference/Post stages
- **Use Case**: ML pipelines

#### Dependency Resolver (`dependencies.py`)
- **Algorithm**: Recursive import resolution + topological sort
- **Scope**: Local imports only (excludes stdlib/third-party)

#### Execution Tracer (`tracer.py`)
- **Method**: `sys.settrace()` for runtime analysis
- **Purpose**: Prune unused functions

---

### 3️⃣ IR Layer (`src/core/intermediate/`)

**Language-neutral intermediate representation**

#### Operation Types (`schema.py`)
| Type | Description | Example |
|------|-------------|---------|
| `FUNCTION_CALL` | Function invocation | `cv2.imread(path)` |
| `METHOD_CALL` | Object method call | `img.astype(np.float32)` |
| `ARITHMETIC` | Binary operations | `img / 255.0` |
| `ASSIGNMENT` | Variable assignment | `result = img` |
| `CONDITIONAL` | if/else statements | `if condition: ...` |
| `LOOP` | for/while loops | `for i in range(10)` |

#### Builder (`builder.py`)
- **Input**: Python AST
- **Output**: `IRPipeline` (inputs, operations, outputs)
- **Features**: Type inference integration

---

### 4️⃣ Mapping Layer (`src/core/mapping/`)

**Python → C++ function mappings**

#### Database (`database.py`)
- **Source**: `config/mappings/*.yaml` (auto-discovery)
- **Implementations**: `config/implementations/*.yaml`
- **Built-in**: 50+ function mappings

#### Mapping Types

| Type | Description | Example |
|------|-------------|---------|
| **1:1 Standard** | Direct mapping | `cv2.imread` → `img::imread` |
| **Custom Template** | Parameterized | `img.astype({dtype})` |
| **N:M Statements** | Multi-statement | Resize + normalize |
| **Inline Implementation** | Full C++ code from YAML | See `img.yaml` |

#### Priority Order
1. **inline_impl** (highest) - Full implementation from YAML
2. **statements** - Multi-statement generation
3. **custom_template** - Single expression template
4. **Standard 1:1** (lowest) - Direct function call

---

### 5️⃣ Generation Layer (`src/core/generation/`)

#### Generator (`generator.py`)
**Main orchestrator for C++ code generation**

**Methods**:
- `generate()`: Single-function project
- `generate_multi_function()`: Multi-function project (main use case)
- `generate_report()`: Conversion summary

**Output**:
- C++ source file
- CMakeLists.txt
- README.md
- Header files (stb_image.h, validator.h)

#### Template Engine (`template.py`)
- **Engine**: Jinja2
- **Custom Filters**: `cpp_type`, `arithmetic_op`, `format_args`
- **Templates**: `base.cpp.j2`, `multi.cpp.j2`

#### LLM Provider (`llm_provider.py`)
**Optional LLM integration for unmapped operations**

| Provider | Model | Use Case |
|----------|-------|----------|
| OpenAI | gpt-4o-mini | Default |
| Anthropic | claude-sonnet-4 | Alternative |
| Fallback | Both | Try OpenAI → Anthropic |

---

### 6️⃣ Validation Layer (`src/core/validation/`)

**Automatic build and result comparison**

#### Executor (`executor.py`)
- `PythonRunner`: Execute Python function, save .npy
- `CppRunner`: Build with CMake, run executable
- **Warmup**: Both runners use warmup iterations

#### Comparator (`comparator.py`)
- **Method**: NumPy `allclose()` with tolerance
- **Metrics**: Max/mean abs/rel differences
- **Output**: Formatted comparison table

**Typical Accuracy**: 77% (with resize), 100% (exact operations)

---

## 🔄 Data Flow

### Standard Conversion

```
┌─────────────────────────────────────────────────────────────┐
│ Python Source Code                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ AST Parsing (parser.py)                                     │
│ • Extract functions, imports, control flow                  │
│ • Build FunctionInfo objects                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Type Inference (inferencer.py)                              │
│ • Annotations → Literals → Library signatures               │
│ • Build type context for each function                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ IR Generation (builder.py)                                  │
│ • Convert AST to IRPipeline                                 │
│ • Extract inputs, operations, outputs                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Mapping (database.py)                                       │
│ • Load YAML mappings                                        │
│ • Map IR operations to C++ equivalents                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Code Generation (generator.py + templates)                  │
│ • Render Jinja2 templates                                   │
│ • Inject implementations from YAML                          │
│ • Generate CMakeLists.txt, README.md                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Build & Validation (executor.py, comparator.py)             │
│ • Build with CMake                                          │
│ • Run Python and C++ versions                               │
│ • Compare results with NumPy                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Design Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Pipeline** | Overall architecture | Analysis → IR → Mapping → Generation |
| **Visitor** | `parser.py` | AST traversal |
| **Strategy** | `llm_provider.py` | Swappable LLM providers |
| **DTO** | `schema.py` | Language-neutral data structures |
| **Template Method** | `generator.py` | Jinja2-based code generation |
| **Factory** | `database.py` | Mapping creation from YAML |

---

## 🔌 Extension Points

### Adding New Library Mappings

**Step 1**: Create mapping file
```yaml
# config/mappings/torch.yaml
functions:
  - python_lib: torch
    python_func: tensor
    cpp_lib: torch
    cpp_func: from_blob
    cpp_headers: ["<torch/torch.h>"]
```

**Step 2**: Auto-discovered on next run

### Adding New Implementations

**Step 1**: Add to implementations YAML
```yaml
# config/implementations/audio.yaml
audio_load: |
  inline AudioData load(const std::string& path) {
      // Implementation
      return data;
  }
```

**Step 2**: Reference in mapping
```yaml
# config/mappings/librosa.yaml
functions:
  - python_lib: librosa
    python_func: load
    cpp_lib: audio
    cpp_func: load
    inline_impl: "audio_load"
```

### Adding New LLM Provider

```python
# src/core/generation/llm_provider.py

class MyLLMGenerator(LLMCodeGenerator):
    def _call_llm(self, prompt: str) -> str | None:
        # Custom implementation
        pass
```

---

## 📊 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Build Time** | ~2-3 seconds | Header-only, selective inclusion |
| **Validation Accuracy** | 77% with resize, 100% exact | Interpolation differences |
| **Type Coverage** | 85% auto-inferred | 15% use `auto` |
| **Function Limit** | < 50 lines | All 276 functions comply |
| **Memory** | < 100 MB | IR generation |

---

## 🛠️ Code Quality Standards

**Enforced by CLAUDE.md**:

✅ Single Responsibility Principle (SRP)
✅ Function length < 50 lines (signature + docstring excluded)
✅ Self-documenting code (minimal inline comments)
✅ Type hints (Python 3.10+ syntax)
✅ No code duplication (DRY)

**Statistics**:
- Total functions: 276
- Violations: 0
- Average function length: ~15 lines

---

## 📦 Dependencies

### Python
- **Core**: Python 3.10+, Jinja2, PyYAML
- **Optional**: openai, anthropic (for LLM features)
- **Validation**: numpy, opencv-python, Pillow

### C++
- **Compiler**: C++17 (GCC 9+, Clang 10+)
- **Build**: CMake 3.15+
- **Runtime**: Header-only (no external dependencies)

---

**Version**: v1.3
**Last Updated**: 2025-12-02
