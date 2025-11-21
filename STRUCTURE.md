# Project Architecture

## Overview

**Architecture Pattern**: Pipeline (Analysis → Mapping → Generation)

## Directory Structure

```
preprocess_python2cpp/
├── src/                          # Source code
│   ├── cli/                      # Command-line interface
│   │   └── main.py               # Entry point, CLI argument handling
│   ├── core/                     # Core functionality
│   │   ├── analysis/             # Python code analysis
│   │   │   ├── parser.py         # AST parsing
│   │   │   ├── inferencer.py    # Type inference
│   │   │   ├── separator.py     # Pipeline separation (Pre/Inf/Post)
│   │   │   ├── dependencies.py  # Dependency resolution
│   │   │   └── tracer.py        # Execution tracing
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
│       │   ├── base.cpp.j2       # Main C++ template
│       │   ├── pipeline_main.cpp.j2  # Pipeline orchestrator
│       │   ├── inference_stub.cpp.j2 # Inference placeholder
│       │   ├── component.h.j2    # Component header
│       │   └── component.cpp.j2  # Component implementation
│       ├── cmake/                # CMake templates
│       │   ├── cmakelists.txt.j2
│       │   └── pipeline_cmakelists.txt.j2
│       └── headers/              # Header-only libraries
│           ├── image.h.j2        # Image processing (JPEG/PNG via stb)
│           ├── stb_image.h       # stb image loader (third-party)
│           ├── stb_image_write.h # stb image writer (third-party)
│           └── validator.h.j2    # Validation utilities
├── examples/                     # Example Python code
│   ├── vision/                   # Image processing examples
│   └── pipeline_demo.py          # Pipeline separation demo
├── tests/                        # Test suite
│   └── test_pipeline.py          # Pipeline separation tests
├── config/                       # Configuration files
│   └── data/                     # Test data
└── .build/                       # Generated output (gitignored)
    └── output/                   # Generated C++ projects
```

## Core Components

### 1. CLI Layer (`src/cli/main.py`)

Entry point and workflow orchestration.

### 2. Analysis Layer (`src/core/analysis/`)

#### 2.1 Parser (`parser.py`)
AST parsing using Visitor Pattern.

#### 2.2 Type Inferencer (`inferencer.py`)
Static type inference via annotations, literals, and library signatures.

#### 2.3 Pipeline Separator (`separator.py`)
Splits code at `# @inference` marker into Pre/Inf/Post stages.

#### 2.4 Dependency Resolver (`dependencies.py`)
Recursive import resolution with topological sorting.

#### 2.5 Execution Tracer (`tracer.py`)
Traces execution path using `sys.settrace` to prune unused code.

### 3. IR (`src/core/intermediate/schema.py`)
Language-neutral DTO. Types: `FUNCTION_CALL`, `METHOD_CALL`, `ARITHMETIC`, `ASSIGNMENT`

### 4. Mapping (`src/core/mapping/`)
Built-in: cv2, numpy (20+ functions). LLM fallback → `learned_mappings.json`

### 5. Generation (`src/core/generation/`)
Jinja2 templates. Modes: Single module or Pipeline. LLM: OpenAI, Anthropic, Vertex AI

### 6. Validation (`src/core/validation/`)
Compares `.npy` results. Accuracy: 100% (no resize) | ~77% (with resize)

### 7. Templates (`src/templates/`)
`image.h`: Header-only (JPEG/PNG via stb_image). `CMake`: Pure C++17

## Data Flow

**Standard**: Parse → Infer Types → Build IR → Map → Generate C++ → Validate

**Pipeline Mode**: Same + Split by `# @inference` → Generate Pre/Inf/Post components

### Variable Reassignment

Jinja2 tracks declared variables: first occurrence uses `auto`, subsequent use assignment.

## Design Patterns

- **Pipeline**: Analysis → Mapping → Generation
- **Visitor**: AST traversal
- **Strategy**: Swappable LLM providers
- **DTO**: IR schema

## Extension Points

1. **New Operations**: Add to `database.py` + implement in `image.h.j2`
2. **LLM Provider**: Inherit from `LLMProvider`
3. **IR Operation**: Update `OperationType` enum + templates

## Dependencies

**Python**: 3.10+, Jinja2, anthropic[vertex] (optional)
**C++**: C++17 compiler, CMake 3.15+
**Runtime**: Header-only, no external C++ dependencies
