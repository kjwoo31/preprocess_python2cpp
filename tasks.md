# Development Roadmap

## Recent Achievements (v0.93)

- ✅ CLAUDE.md standards compliance (SRP, 50-line limit, type hints)
- ✅ Code deduplication (~50 lines removed)
- ✅ CLAUDE.md work completion guidelines added
- ✅ Bug fix: type checking in list elements
- ✅ Removed unused code (parser._current_function)

## Priority Tasks

### 🔴 P1: Pipeline Separation & Multi-file Support ✅

**Core Analysis**:
- [x] `src/core/analysis/separator.py`: Split Pre/Inf/Post by `# @inference` comment
- [x] `src/core/analysis/dependencies.py`: Recursive import resolution
- [x] `src/core/analysis/tracer.py`: Execution path tracking with `sys.settrace`

**Code Generation**:
- [x] Update `generator.py`: `generate_pipeline()` method for multi-module output
- [x] Template `pipeline_main.cpp.j2`: Orchestrate Pre → Inf → Post
- [x] Template `inference_stub.cpp.j2`: Placeholder for inference block
- [x] Template `component.h.j2` & `component.cpp.j2`: Component modules
- [x] Template `pipeline_cmakelists.txt.j2`: Pipeline CMake config

**CLI Updates**:
- [x] Add `--pipeline` flag for split conversion mode
- [x] Add `--recursive` flag for dependency resolution
- [x] Change `--validate` to default true, add `--no-validate`
- [x] Implement pipeline conversion workflow

**Testing & Examples**:
- [x] `tests/test_pipeline.py`: Pipeline separation tests
- [x] `examples/pipeline_demo.py`: Image classification demo

### 🟡 P2: Image Processing Enhancements

**Additional Operations**:
- [ ] Canny edge detection
- [ ] Sobel filter
- [ ] Morphological operations (erode, dilate)
- [ ] Full bilateral filter implementation

### 🔵 P2.5: Declarative Function Mapping Configuration (Refactoring)

**Goal**: Move hardcoded function mappings (in `database.py`) to an external configuration file (YAML/JSON) to make adding new rules more intuitive and data-driven.

**Tasks**:
- [ ] Design YAML/JSON schema for function/constant mappings
- [ ] Migrate hardcoded mappings (OpenCV, NumPy, Librosa) to config files
- [ ] Refactor `MappingDatabase` to load from config
- [ ] Add schema validation and documentation
- [ ] **Support N:M mapping (Complex Patterns)**: Allow one Python function to map to multiple C++ statements (e.g., `cv2.split` -> `std::vector<cv::Mat> ch; cv::split(src, ch);`)

### 🟢 P3: Advanced Features

**Control Flow Support**:
- [ ] Extend IR schema for control flow
- [ ] Parse if/else statements
- [ ] Parse for/while loops
- [ ] Generate C++ control flow code

**Method Improvements**:
- [ ] Support chained calls: `a.method1().method2()`
- [ ] Standalone `astype()` handling
- [ ] NumPy array methods

## Known Limitations

- **Control flow**: Limited if/else and loop support
- **Complex operations**: Some Python operations (np.argmax) may not map perfectly
- **LLM integration**: Requires GCP Vertex AI access (optional)
- **Type inference**: Sometimes uses `auto` instead of concrete types
- **Method chains**: Limited support
- **Bilateral filter**: Simplified stub implementation

## Progress

| Component | Status | % |
|-----------|--------|---|
| Python AST Parsing | ✅ Done | 90% |
| Type Inference | ✅ Done | 85% |
| IR Generation | ✅ Done | 85% |
| Mapping Database | ✅ Done | 80% |
| Code Generation | ✅ Done | 90% |
| Pipeline Separation | ✅ Done | 100% |
| Dependency Resolution | ✅ Done | 100% |
| Execution Tracing | ✅ Done | 100% |
| Header-Only Library | ✅ Done | 100% |
| Build System | ✅ Done | 100% |
| CLI | ✅ Done | 100% |
| Validation | ✅ Done | 100% |
| LLM Integration | ⚠️ Blocked | 50% |
| Control Flow | ❌ Pending | 0% |
| **Overall** | **~93%** | **93%** |

---

**Last Updated**: 2025-11-22 (v0.93)
