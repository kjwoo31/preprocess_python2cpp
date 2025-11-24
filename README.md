# Python-to-C++ Porting Agent

Automatically converts Python data preprocessing code to pure C++ with header-only libraries.

## Features

- ✅ **1-to-1 File Mapping**: One Python file → One C++ file (all functions converted)
- ✅ **Minimal Dependencies**: Header-only libraries (stb_image included)
- ✅ **Auto-Validation**: Python vs C++ result comparison (default enabled)
- ✅ **Complete Projects**: Generates C++ source + CMakeLists.txt

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, C++ compiler (GCC 9+), CMake 3.15+

## Quick Start

```bash
# Prepare test image
python3 -c "from PIL import Image; import numpy as np; Image.fromarray(np.random.randint(0,255,(100,100,3),dtype=np.uint8)).save('test.jpg')"

# Convert Python to C++ (all functions in one file)
python3 src/cli/main.py -i examples/vision/image_preprocessing.py --test-input test.jpg

# Generated: .build/output/image_preprocessing/
# - image_preprocessing.cpp (contains 2 functions from __main__ block)
# - CMakeLists.txt
# - README.md

# Build and run
cd .build/output/image_preprocessing
mkdir build && cd build
cmake .. && make

# Run (executes code from Python's if __name__ == "__main__")
./image_preprocessing ../../../../test.jpg
# Output:
# Processing image: ../../../../test.jpg
# preprocess_image: shape=(224, 224, 3), dtype=float32
# preprocess_with_color_conversion: shape=(256, 256, 3), dtype=float32

# Note: All statements from Python's __main__ block (variables, assignments, function calls)
# are converted to C++. Only functions called in __main__ are defined - unused functions are skipped.

# Validation: ~81% match (bilinear interpolation differences)
```

## CLI Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Python source file |
| `-f, --function` | Function name to convert (optional) |
| `--test-input` | Test input for validation |
| `--no-validate` | Skip validation |
| `-v, --verbose` | Verbose output |

## Documentation

- **STRUCTURE.md** - Architecture and code organization
- **tasks.md** - Development roadmap
- **CLAUDE.md** - Coding standards

## License

MIT License
