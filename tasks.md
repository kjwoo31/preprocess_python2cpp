# Development Roadmap

## Recent Achievements (v0.89)

- ✅ stb_image.h integration (JPEG, PNG, BMP, TGA support)
- ✅ Auto-detection: stb for JPEG/PNG, native for PPM/PGM
- ✅ Type system corrected: Image (uint8) + ImageF (float32) classes
- ✅ Resize bug fixed: OpenCV coordinate formula implementation
- ✅ Validation: 82.80% exact match, 100% within 1% error
- ✅ Performance report: Comprehensive testing of all examples
- ✅ CLAUDE.md compliance: All functions under 50 lines

## Priority Tasks

### 🟡 P2: Image Processing Enhancements

**Additional Operations**:
- [ ] Canny edge detection
- [ ] Sobel filter
- [ ] Morphological operations (erode, dilate)
- [ ] Full bilateral filter implementation

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

- Control flow: Limited if/else and loop support
- LLM integration: Requires GCP Vertex AI access (optional)
- Type inference: Sometimes uses `auto` instead of concrete types
- Method chains: Limited support
- Bilateral filter: Simplified stub implementation

## Progress

| Component | Status | % |
|-----------|--------|---|
| Python AST Parsing | ✅ Done | 90% |
| Type Inference | ✅ Done | 85% |
| IR Generation | ✅ Done | 85% |
| Mapping Database | ✅ Done | 80% |
| Code Generation | ✅ Done | 85% |
| Header-Only Library | ✅ Done | 100% |
| Build System | ✅ Done | 100% |
| CLI | ✅ Done | 100% |
| Validation | ✅ Done | 100% |
| LLM Integration | ⚠️ Blocked | 50% |
| Control Flow | ❌ Pending | 0% |
| **Overall** | **~89%** | **89%** |

---

**Last Updated**: 2025-11-20
