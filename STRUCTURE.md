# Project Architecture

## Overview

Python-to-C++ converter with dependency-free architecture. Converts Python preprocessing code to standalone C++ projects with header-only libraries.

**Architecture Pattern**: Pipeline Architecture (Analysis → Mapping → Generation)

## Directory Structure

```
preprocess_python2cpp/
├── src/                          # Source code
│   ├── cli/                      # Command-line interface
│   │   └── main.py               # Entry point, CLI argument handling
│   ├── core/                     # Core functionality
│   │   ├── analysis/             # Python code analysis
│   │   │   ├── parser.py         # AST parsing
│   │   │   └── inferencer.py    # Type inference
│   │   ├── intermediate/         # IR (Intermediate Representation)
│   │   │   └── schema.py         # IR data structures
│   │   ├── mapping/              # Python-to-C++ mappings
│   │   │   ├── core.py           # Mapping core types
│   │   │   └── database.py      # Function mapping database
│   │   ├── generation/           # C++ code generation
│   │   │   ├── template.py       # Jinja2 template engine
│   │   │   ├── filters.py        # Custom Jinja2 filters
│   │   │   ├── generator.py     # Code generation orchestrator
│   │   │   └── llm_provider.py  # LLM integration (optional)
│   │   └── validation/           # Auto-validation
│   │       ├── executor.py       # Build/run executor
│   │       └── comparator.py    # Result comparison
│   └── templates/                # Jinja2 templates
│       ├── cpp/                  # C++ code templates
│       │   └── base.cpp.j2       # Main C++ template
│       ├── cmake/                # CMake templates
│       │   └── cmakelists.txt.j2
│       └── headers/              # Header-only libraries
│           ├── image.h.j2        # Image processing (JPEG/PNG via stb)
│           ├── stb_image.h       # stb image loader (third-party)
│           ├── stb_image_write.h # stb image writer (third-party)
│           └── validator.h.j2    # Validation utilities
├── examples/                     # Example Python code
│   └── vision/                   # Image processing examples
├── config/                       # Configuration files
│   └── data/                     # Test data
└── .build/                       # Generated output (gitignored)
    └── output/                   # Generated C++ projects
```

## Core Components

### 1. CLI Layer (`src/cli/main.py`)

**Responsibility**: User interface and workflow orchestration

**Key Functions**:
- `main()` - Entry point, argument parsing
- `build_ir_pipeline()` - Creates IR from Python AST
- `_extract_operations()` - Converts AST to IR operations
- `_create_binary_operation()` - Handles arithmetic operations
- `_create_function_call_operation()` - Handles function/method calls

### 2. Analysis Layer (`src/core/analysis/`)

#### 2.1 Parser (`parser.py`)

**Responsibility**: Python AST parsing and structural analysis

**Classes**:
- `ImportInfo` - Import statement metadata
- `FunctionInfo` - Function definition metadata
- `PythonASTParser` - Main parser class

**Key Methods**:
- `parse(source_code)` - Parse Python code into AST
- `get_library_usage()` - Extract imported libraries
- `get_function_calls()` - Extract all function calls

**Design Pattern**: Visitor Pattern (uses `ast.NodeVisitor`)

#### 2.2 Type Inferencer (`inferencer.py`)

**Responsibility**: Static type inference for Python variables

**Inference Strategies**:
1. Type annotations (explicit)
2. Literal value analysis
3. Function return type tracking
4. Library-specific knowledge (cv2, numpy, PIL)

**Key Methods**:
- `infer_type(node, context)` - Main inference entry point
- `_infer_call_type()` - Infer from function calls
- `_infer_binop_type()` - Infer from binary operations
- `update_context()` - Track variable types through code flow

### 3. Intermediate Representation (`src/core/intermediate/schema.py`)

**Responsibility**: Language-neutral IR data structures

**Core Classes**:
- `TypeHint` - Type information with C++ mapping
- `IROperation` - Single operation (function call, arithmetic, etc.)
- `IRInput` - Function input parameter
- `IROutput` - Function return value
- `IRPipeline` - Complete conversion pipeline

**Operation Types** (`OperationType` enum):
- `FUNCTION_CALL` - Library function calls (cv2.imread)
- `METHOD_CALL` - Object method calls (img.astype)
- `ARITHMETIC` - Binary operations (+, -, *, /)
- `ASSIGNMENT` - Variable assignments
- `CONTROL_FLOW` - If/else, loops (future)

**Design Pattern**: Data Transfer Object (DTO)

### 4. Mapping Layer (`src/core/mapping/`)

#### 4.1 Core (`core.py`)

**Responsibility**: Python-to-C++ mapping types

**Classes**:
- `FunctionMapping` - Maps Python function to C++ equivalent
- `OperatorMapping` - Maps Python operators to C++
- `TypeMapping` - Maps Python types to C++

#### 4.2 Database (`database.py`)

**Responsibility**: Function mapping database and LLM integration

**Built-in Mappings** (20+ functions):
- **OpenCV**: imread, resize, cvtColor, GaussianBlur, bilateralFilter
- **NumPy**: zeros, ones, array, mean, std, astype, reshape, transpose
- **Operators**: +, -, *, /, %

**Features**:
- `get_mapping()` - Retrieve mapping for Python function
- `add_learned_mapping()` - Save LLM-generated mappings
- `auto_load_learned()` - Load previously learned mappings
- LLM fallback for unmapped functions

**Storage**: `learned_mappings.json` (auto-saved)

### 5. Generation Layer (`src/core/generation/`)

#### 5.1 Template Engine (`template.py`)

**Responsibility**: Jinja2 template rendering

**Templates Rendered**:
- C++ source code (`base.cpp.j2`)
- CMakeLists.txt (`cmakelists.txt.j2`)
- Header libraries (`image.h.j2`, `validator.h.j2`)
- README.md for generated project

**Custom Jinja2 Filters** (`filters.py`):
- `cpp_type` - Convert Python type to C++
- `arithmetic_op` - Convert operator names
- `format_args` - Format function arguments
- `cpp_lib` - Convert Python library names

#### 5.2 Generator (`generator.py`)

**Responsibility**: Orchestrate C++ project generation

**Generation Steps**:
1. Create output directory
2. Map operations to C++ code
3. Query LLM for unmapped operations (optional)
4. Render C++ source from template
5. Generate CMakeLists.txt
6. Copy header libraries
7. Generate README.md

**Key Methods**:
- `generate(pipeline)` - Main generation entry point
- `_map_operations()` - Map IR operations to C++
- `_query_llm_for_mapping()` - LLM fallback
- `_save_llm_as_mapping()` - Save learned mappings
- `_extract_cpp_lib_func()` - Extract C++ library and function
- `_extract_cpp_headers()` - Extract header includes
- `_create_function_mapping()` - Create mapping from extracted info
- `_copy_stb_headers()` - Copy stb_image headers to output
- `_generate_*()` - Generate individual files

#### 5.3 LLM Provider (`llm_provider.py`)

**Responsibility**: LLM API integration for unmapped operations

**Providers**:
- `OpenAIProvider` - GPT-4 integration
- `AnthropicProvider` - Claude Direct API
- `VertexAIProvider` - Claude via Google Cloud (default)

**Key Methods**:
- `query_for_mapping()` - Query LLM for C++ equivalent
- `_build_prompt()` - Construct LLM prompt (orchestrator)
- `_build_prompt_context()` - Build context section
- `_build_prompt_operation()` - Build operation section
- `_build_prompt_task()` - Build task guidelines
- `_build_prompt_examples()` - Build examples section
- `_parse_response()` - Extract C++ code from LLM response

### 6. Validation Layer (`src/core/validation/`)

#### 6.1 Executor (`executor.py`)

**Responsibility**: Build and execute generated C++ code

**Classes**:
- `ValidationExecutor` - Main executor
- `PythonRunner` - Run Python code for comparison
- `CppRunner` - Build and run C++ code

**Flow**:
1. Run Python function, save results (.npy)
2. Build C++ code (CMake + Make)
3. Run C++ executable, save results (.npy)
4. Compare results (numerical accuracy)

**Validation Accuracy**:
- Functions without resize: 100% match
- Functions with resize: ~77% match
- Functions with bilateralFilter: ~63% match (stub)

#### 6.2 Comparator (`comparator.py`)

**Responsibility**: Compare Python and C++ execution results

**Comparison Metrics**:
- Numerical accuracy (MSE, RMSE)
- Shape matching
- Data type matching
- Performance (execution time)

### 7. Template Files (`src/templates/`)

#### 7.1 C++ Template (`cpp/base.cpp.j2`)

**Features**:
- Variable reassignment tracking (Jinja2 `set`)
- Type-safe code generation
- Dependency-free includes
- `.npy` result saving for validation

#### 7.2 CMake Template (`cmake/cmakelists.txt.j2`)

**Features**:
- Pure C++17 standard
- No external dependencies (no `find_package()`)
- Optimization flags (-O3 -march=native)
- Simple executable build

#### 7.3 Image Header (`headers/image.h.j2`)

**Image Processing Library with stb_image Integration**

**Class**: `Image`
- Data storage: `std::vector<float>`
- Metadata: height, width, channels

**Functions** (all `inline`):
- `imread()` - Auto-detects format (JPEG, PNG, BMP via stb / PPM/PGM fallback)
- `imread_stb()` - Load JPEG, PNG, BMP, TGA using stb_image.h
- `imread_ppm()` - Load PPM/PGM format (dependency-free fallback)
- `get_extension()` - Helper to detect file format
- `resize()` - Bilinear interpolation
- `cvtColor_BGR2RGB()` - Channel swap
- `GaussianBlur()` - 3x3 Gaussian kernel
- `bilateralFilter()` - Stub implementation
- Arithmetic operators (+, -, *, /)

**Image Format Support**:
- JPEG, PNG, BMP, TGA (via stb_image.h)
- PPM/PGM (native implementation)
- Auto-detection based on file extension

**Design**: Header-only library with optional stb dependency

## Data Flow

### Complete Conversion Pipeline

```
1. USER INPUT
   ├─ Python source file
   ├─ Function name
   └─ CLI options (--llm, --validate, --ir-only, -v)

2. ANALYSIS PHASE (src/core/analysis/)
   ├─ PythonASTParser.parse(source_code)
   │   └─ Extract imports, functions, structure
   ├─ TypeInferenceEngine.infer_type()
   │   └─ Infer types for all variables
   └─ OUTPUT: ast.Module + type context

3. IR GENERATION (src/cli/main.py)
   ├─ build_ir_pipeline()
   │   ├─ Extract function inputs
   │   ├─ Extract operations (_extract_operations)
   │   │   ├─ _create_function_call_operation()
   │   │   ├─ _create_binary_operation()
   │   │   └─ Track variable declarations
   │   └─ Extract outputs
   └─ OUTPUT: IRPipeline object

4. MAPPING PHASE (src/core/mapping/)
   ├─ MappingDatabase.get_mapping(lib, func)
   │   ├─ Check built-in mappings
   │   ├─ Check learned mappings
   │   └─ Return FunctionMapping or None
   └─ OUTPUT: Python → C++ mappings

5. CODE GENERATION (src/core/generation/)
   ├─ CodeGenerator.generate(pipeline)
   │   ├─ Create output directory
   │   ├─ Map operations to C++
   │   │   ├─ Use MappingDatabase
   │   │   └─ LLM fallback if unmapped (--llm)
   │   ├─ TemplateEngine.render()
   │   │   ├─ Render C++ source (base.cpp.j2)
   │   │   ├─ Render CMakeLists.txt
   │   │   ├─ Render image.h
   │   │   └─ Render README.md
   │   └─ Save learned mappings
   └─ OUTPUT: Complete C++ project

6. VALIDATION (Optional, src/core/validation/)
   ├─ PythonRunner.run() → results_python.npy
   ├─ CppRunner.build_and_run() → results_cpp.npy
   ├─ ResultComparator.compare()
   └─ OUTPUT: Validation report

7. USER OUTPUT
   └─ .build/output/{function_name}/
       ├── {function_name}.cpp
       ├── image.h
       ├── CMakeLists.txt
       ├── validator.h
       └── README.md
```

### Variable Reassignment Handling

**Problem**: Python allows variable reassignment, C++ requires single declaration

**Solution** (implemented in `base.cpp.j2`):

```jinja2
{% set declared_vars = [] %}
{% for op in pipeline.operations %}
  {% if op.output in declared_vars %}
    {{ op.output }} = ...;           {# Assignment #}
  {% else %}
    auto {{ op.output }} = ...;      {# Declaration #}
    {% set _ = declared_vars.append(op.output) %}
  {% endif %}
{% endfor %}
```

**Example**:
```python
# Python
img = cv2.imread(path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
```

```cpp
// Generated C++
auto img = img::imread(path);      // First: declaration
img = img::resize(img, 224, 224);  // Second: assignment
img = img / 255.0;                 // Third: assignment
```

## Design Patterns Used

1. **Pipeline Architecture**
   - Linear flow: Analysis → Mapping → Generation
   - Each stage independent and testable

2. **Visitor Pattern**
   - AST traversal in `parser.py`
   - Separate logic from data structure

3. **Strategy Pattern**
   - Multiple LLM providers (OpenAI, Anthropic, Vertex)
   - Swappable at runtime

4. **Template Method**
   - `LLMProvider` abstract base class
   - Concrete implementations override specific methods

5. **Data Transfer Object (DTO)**
   - IR schema classes (`IROperation`, `IRPipeline`)
   - Language-neutral data transfer

6. **Singleton-like**
   - `MappingDatabase` maintains single source of truth
   - Learned mappings persisted to JSON

## Extension Points

### Adding New Operations

**Step 1**: Add mapping to `database.py`:
```python
self.add_mapping(FunctionMapping(
    python_lib='cv2',
    python_func='Canny',
    cpp_lib='img',
    cpp_func='Canny',
    notes='Edge detection'
))
```

**Step 2**: Implement in `image.h.j2`:
```cpp
inline Image Canny(const Image& src, double threshold1, double threshold2) {
    // Implementation
}
```

### Adding New LLM Provider

Inherit from `LLMProvider` in `llm_provider.py`:
```python
class NewProvider(LLMProvider):
    def query_for_mapping(self, python_code: str, lib: str, func: str) -> Optional[str]:
        # Implementation
        pass
```

### Adding New IR Operation Type

1. Add to `OperationType` enum in `schema.py`
2. Update `_extract_operations()` in `main.py`
3. Update template in `base.cpp.j2`

## Performance Characteristics

**Analysis Phase**: O(n) where n = AST nodes
**Mapping Phase**: O(1) hash lookup
**Code Generation**: O(m) where m = IR operations
**LLM Queries**: O(k * network_latency) where k = unmapped operations

**Typical Conversion Time**:
- Simple function (3 ops): ~50ms
- Medium function (10 ops): ~100ms
- With LLM (per unmapped op): +2-5s

**Memory Usage**:
- Small function: ~5MB
- Large function: ~20MB
- Primarily AST and IR storage

## Dependencies

**Python Runtime**:
- Python 3.10+ (AST features)
- Jinja2 (template engine)
- anthropic[vertex] (optional, for LLM)

**Build Time**:
- C++17 compiler (GCC 9+, Clang 12+, MSVC 2019+)
- CMake 3.15+

**No Runtime C++ Dependencies**:
- All libraries header-only
- Pure standard library C++17
