# Project Architecture

## Overview

**Architecture Pattern**: Pipeline (Analysis → Mapping → Generation)

**Version**: v1.2 - Code quality standards compliance

## Directory Structure

```
preprocess_python2cpp/
├── README.md                     # User guide and quick start
├── STRUCTURE.md                  # This file - architecture documentation
├── CLAUDE.md                     # Development guidelines
├── tasks.md                      # Development roadmap
│
├── config/                       # ✨ STRUCTURED CONFIGURATION (v1.1)
│   ├── mappings/                 # Python → C++ function mappings
│   │   ├── opencv.yaml           # cv2.* mappings
│   │   ├── numpy.yaml            # numpy.* mappings
│   │   ├── librosa.yaml          # librosa.* mappings
│   │   └── pil.yaml              # PIL.* mappings
│   │
│   ├── implementations/          # C++ inline implementations
│   │   └── img.yaml              # img::* functions (header-free)
│   │
│   └── schema/                   # Schema documentation
│       └── mappings.yaml         # YAML schema and examples
│
├── src/
│   ├── cli/                      # Command-line interface
│   │   └── main.py               # Entry point
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
│   │   │   └── schema.py         # IR data structures
│   │   │
│   │   ├── mapping/              # Python-to-C++ mappings
│   │   │   ├── core.py           # Mapping core types
│   │   │   ├── database.py       # ✨ Loads from config/mappings/
│   │   │   └── validator.py     # YAML validation
│   │   │
│   │   ├── generation/           # C++ code generation
│   │   │   ├── template.py       # Jinja2 template engine
│   │   │   ├── filters.py        # Custom Jinja2 filters
│   │   │   ├── generator.py     # ✨ Passes implementations to templates
│   │   │   └── llm_provider.py  # LLM integration (optional)
│   │   │
│   │   └── validation/           # Auto-validation
│   │       ├── executor.py       # Build/run executor
│   │       └── comparator.py    # Result comparison
│   │
│   └── templates/                # Jinja2 templates
│       ├── cpp/                  # C++ code templates
│       │   ├── base.cpp.j2       # ✨ Inline img namespace
│       │   ├── pipeline_main.cpp.j2  # Pipeline orchestrator
│       │   ├── inference_stub.cpp.j2 # Inference placeholder
│       │   ├── component.h.j2    # Component header
│       │   └── component.cpp.j2  # Component implementation
│       │
│       ├── cmake/                # CMake templates
│       │   ├── cmakelists.txt.j2
│       │   └── pipeline_cmakelists.txt.j2
│       │
│       └── headers/              # Third-party headers
│           ├── stb_image.h       # stb image loader
│           ├── stb_image_write.h # stb image writer
│           └── validator.h.j2    # Validation utilities
│
├── examples/                     # Example Python code
│   └── vision/                   # Image processing examples
│
├── tests/                        # Test suite
│   └── test_pipeline.py          # Pipeline separation tests
│
└── .build/                       # Generated output (gitignored)
    └── output/                   # Generated C++ projects
```

## Core Components

### 1. CLI Layer (`src/cli/main.py`)

Entry point and workflow orchestration.

**Key flags**:
- `--output`: Output directory
- `--pipeline`: Enable pipeline separation mode
- `--validate`: Auto-validate output (default: true)

### 2. Analysis Layer (`src/core/analysis/`)

#### 2.1 Parser (`parser.py`)
AST parsing using Visitor Pattern. Extracts functions, types, control flow.

#### 2.2 Type Inferencer (`inferencer.py`)
Static type inference via annotations, literals, and library signatures.

#### 2.3 Pipeline Separator (`separator.py`)
Splits code at `# @inference` marker into Pre/Inf/Post stages.

#### 2.4 Dependency Resolver (`dependencies.py`)
Recursive import resolution with topological sorting.

#### 2.5 Execution Tracer (`tracer.py`)
Traces execution path using `sys.settrace` to prune unused code.

### 3. IR Layer (`src/core/intermediate/schema.py`)

Language-neutral intermediate representation.

**Operation Types**:
- `FUNCTION_CALL`: Function invocations
- `METHOD_CALL`: Object method calls
- `ARITHMETIC`: Binary operations
- `ASSIGNMENT`: Variable assignment
- `CONDITIONAL`: if/else statements (v1.0+)
- `LOOP`: for/while loops (v1.0+)

### 4. Mapping Layer (`src/core/mapping/`)

#### 4.1 Database (`database.py`)

**v1.1 Changes**:
- Loads mappings from `config/mappings/*.yaml` (auto-discovery)
- Loads implementations from `config/implementations/*.yaml`
- Stores implementations in memory: `Dict[str, str]`

**Mapping Types**:
1. **Standard 1:1**: Direct function mapping
2. **Custom Template**: Parameterized code generation
3. **N:M Mapping**: One Python call → multiple C++ statements
4. **Inline Implementation** (v1.1): Complete C++ code from YAML

**Built-in mappings**: cv2, numpy, librosa, PIL (50+ functions)

#### 4.2 Validator (`validator.py`)

Schema validation for YAML configuration files.

### 5. Generation Layer (`src/core/generation/`)

#### 5.1 Generator (`generator.py`)

**v1.1 Changes**:
- Disabled `_generate_image_header()` (no longer generates image.h)
- Passes `implementations` dict to template context
- Selective code injection (only used functions)

**Modes**:
- **Single**: One C++ file + CMakeLists.txt
- **Pipeline**: Pre/Inf/Post components + orchestrator

#### 5.2 Template Engine (`template.py`)

Jinja2-based code generation with custom filters.

**Key Templates**:
- `base.cpp.j2`: Inline img namespace + main function
- `pipeline_main.cpp.j2`: Multi-component orchestrator

### 6. Validation Layer (`src/core/validation/`)

Automatic build and result comparison.

**Accuracy**: 77% match (with resize), 100% (without resize)

## Configuration System (v1.1)

### Structure

```
config/
├── mappings/           # Function mappings
│   ├── opencv.yaml
│   ├── numpy.yaml
│   ├── librosa.yaml
│   └── pil.yaml
│
├── implementations/    # C++ code snippets
│   └── img.yaml
│
└── schema/             # Documentation
    └── mappings.yaml
```

### Mapping Format

```yaml
functions:
  - python_lib: cv2
    python_func: resize
    cpp_lib: img
    cpp_func: resize
    cpp_headers: []
    is_method: false
    inline_impl: "img_resize"  # References implementations/img.yaml
    notes: "Resizes image"
```

### Implementation Format

```yaml
img_resize: |
  inline Image resize(const Image& src, int new_height, int new_width) {
      // ... complete implementation ...
      return dst;
  }
```

### Priority Order

When multiple mapping methods specified:
1. **inline_impl** (highest) - Loads from `implementations/*.yaml`
2. **statements** - N:M multi-statement generation
3. **custom_template** - Single expression template
4. **Standard 1:1** (lowest) - Direct function call

## Inline Implementation System (v1.1)

### Architecture

**No image.h dependency**: Function implementations stored in YAML and injected directly into generated C++ code.

**Benefits**:
- ✅ No external header files
- ✅ Faster builds (only includes used functions)
- ✅ Maintainable (central YAML repository)
- ✅ Declarative (mappings reference implementations by name)

### Code Generation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ config/mappings/opencv.yaml                                 │
│   python_func: resize                                       │
│   inline_impl: "img_resize"                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ├──→ database.py loads mapping
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ config/implementations/img.yaml                             │
│   img_resize: |                                             │
│     inline Image resize(...) { ... }                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ├──→ database.py loads implementation
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ generator.py → base.cpp.j2                                  │
│   - Injects only implementations used by pipeline           │
│   - Generates inline img namespace                          │
└─────────────────────────────────────────────────────────────┘
```

### Generated Code Example

```cpp
namespace img {
    struct Image { /* ... */ };
    struct ImageF { /* ... */ };

    // Only implementations used by this pipeline
    inline Image resize(const Image& src, int new_height, int new_width) {
        // Injected from implementations/img.yaml
    }

    inline Image imread(const std::string& path, int mode = 1) {
        // Injected from implementations/img.yaml
    }
}

auto preprocess(const std::string& path) {
    auto img_001 = img::imread(path, 1);
    auto img_002 = img::resize(img_001, 224, 224);
    return img_002;
}
```

### Adding New Functions

**Step 1**: Add implementation to `config/implementations/img.yaml`

```yaml
img_my_function: |
  inline Image my_function(const Image& src, int param) {
      // ... implementation ...
      return dst;
  }
```

**Step 2**: Add mapping to `config/mappings/opencv.yaml`

```yaml
- python_lib: cv2
  python_func: my_function
  cpp_lib: img
  cpp_func: my_function
  inline_impl: "img_my_function"
```

**Step 3**: Done! Generator automatically includes it when used.

## Data Flow

### Standard Mode

```
Python Code
    ↓
AST Parsing (parser.py)
    ↓
Type Inference (inferencer.py)
    ↓
IR Generation (schema.py)
    ↓
Function Mapping (database.py)
    ↓
C++ Code Generation (generator.py + templates)
    ↓
Build & Validate (validation/)
```

### Pipeline Mode

```
Python Code with # @inference marker
    ↓
Pipeline Separation (separator.py)
    ↓
Pre/Inf/Post Analysis
    ↓
IR Generation for each component
    ↓
Multi-component C++ Generation
    ↓
Build & Validate
```

## Design Patterns

- **Pipeline**: Analysis → Mapping → Generation
- **Visitor**: AST traversal in parser
- **Strategy**: Swappable LLM providers
- **DTO**: IR schema for language neutrality
- **Template Method**: Code generation via Jinja2

## Extension Points

### 1. New Library Mappings

Add files to `config/mappings/`:
```bash
config/mappings/torch.yaml  # Auto-discovered
```

### 2. New Implementations

Add to `config/implementations/`:
```bash
config/implementations/audio.yaml  # For audio processing
```

### 3. New LLM Provider

Inherit from `LLMProvider` in `llm_provider.py`

### 4. New IR Operation

Update `OperationType` enum + add template rendering logic

## Dependencies

**Python**: 3.10+, Jinja2, PyYAML, anthropic[vertex] (optional)

**C++**: C++17 compiler, CMake 3.15+

**Runtime**: Header-only, no external C++ dependencies

## Performance Characteristics

- **Build Time**: ~2-3 seconds (header-only, selective inclusion)
- **Validation Accuracy**: 77% (with resize), 100% (exact operations)
- **Type Coverage**: 85% (auto-inferred), 15% (manual annotation)

## Code Quality

**CLAUDE.md Compliance** (v1.2):
- All functions under 50-line limit
- Self-documenting code (minimal comments)
- Strong type hints (Python 3.10+ syntax)
- Single Responsibility Principle enforced

---

**Last Updated**: 2025-11-22 (v1.2)
