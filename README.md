# Python-to-C++ Porting Agent

Automatically converts Python data preprocessing code to optimized C++ with header-only libraries.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **1:1 File Mapping** | One Python file → One C++ project (all functions) |
| **Minimal Dependencies** | Header-only libraries (stb_image included) |
| **Auto-Validation** | Python vs C++ result comparison (enabled by default) |
| **Complete Projects** | Generates C++ source + CMakeLists.txt + README |
| **Code Quality** | All functions < 50 lines, follows SRP |

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, C++ compiler (GCC 9+), CMake 3.15+

### Basic Usage

```bash
# 1. Generate test image
python3 -c "from PIL import Image; import numpy as np; \
Image.fromarray(np.random.randint(0,255,(100,100,3),dtype=np.uint8)).save('test.jpg')"

# 2. Convert Python to C++
python3 src/cli/main.py -i examples/vision/image_preprocessing.py --test-input test.jpg

# Generated output:
# .build/output/image_preprocessing/
#   ├── image_preprocessing.cpp  (2 functions from __main__ block)
#   ├── CMakeLists.txt
#   └── README.md

# 3. Build and run C++
cd .build/output/image_preprocessing
mkdir build && cd build
cmake .. && make

# 4. Execute
./image_preprocessing ../../../../test.jpg
```

**Output Example**:
```
Processing image: ../../../../test.jpg
preprocess_image: shape=(224, 224, 3), dtype=float32
preprocess_with_color_conversion: shape=(256, 256, 3), dtype=float32
```

---

## 📋 CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Python source file | **Required** |
| `--test-input` | Test input for validation | Optional |
| `-f, --function` | Specific function to convert | All functions |
| `--no-validate` | Skip validation | Validate |
| `-v, --verbose` | Verbose output | Off |

### Examples

```bash
# Convert specific function
python3 src/cli/main.py -i examples/vision/mnist_inference.py -f mnist_inference

# Skip validation
python3 src/cli/main.py -i examples/vision/image_preprocessing.py --no-validate

# Verbose mode
python3 src/cli/main.py -i examples/vision/image_preprocessing.py -v --test-input test.jpg
```

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| **[STRUCTURE.md](STRUCTURE.md)** | Architecture and code organization |
| **[tasks.md](tasks.md)** | Development roadmap and progress |
| **[CLAUDE.md](CLAUDE.md)** | Coding standards and guidelines |

---

## 🔧 How It Works

```
┌─────────────────┐
│  Python Code    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AST Parsing    │  Extract functions, types, control flow
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Type Inference │  Static analysis, library signatures
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  IR Generation  │  Language-neutral intermediate representation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Mapping        │  Python → C++ function mapping (YAML-based)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Code Gen       │  C++ source + CMakeLists.txt + headers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validation     │  Build, run, compare results
└─────────────────┘
```

---

## 🎓 Supported Libraries

| Python Library | C++ Equivalent | Functions |
|----------------|----------------|-----------|
| **cv2** | `img::` namespace | imread, resize, cvtColor, etc. |
| **numpy** | `cv::Mat` / Eigen | array ops, reshape, transpose |
| **PIL** | `stb_image` | Image.open, save |

Full mapping: `config/mappings/*.yaml`

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Build Time** | ~2-3 seconds (header-only) |
| **Validation Accuracy** | 77% (with resize), 100% (exact ops) |
| **Type Coverage** | 85% auto-inferred |
| **Code Quality** | All functions < 50 lines |

---

## 🐛 Troubleshooting

**Build fails?**
```bash
# Check CMake version
cmake --version  # Requires 3.15+

# Check compiler
g++ --version    # Requires GCC 9+
```

**Validation mismatch?**
- Resize operations use different interpolation (bilinear)
- Floating-point precision differences are normal (~1e-5)

**Missing functions?**
- Only functions called in `if __name__ == "__main__"` are converted
- Use `-f function_name` to convert specific functions

---

## 📜 License

MIT License

---

**Version**: v1.3 | **Last Updated**: 2025-12-02
