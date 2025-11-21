# Python-to-C++ Porting Agent

Automatically converts Python data preprocessing code to pure C++ with header-only libraries.

## Features

- ✅ **Minimal Dependencies**: Header-only libraries (stb_image included)
- ✅ **Pipeline Separation**: Split Pre/Inf/Post with `# @inference` marker
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

# Convert Python to C++ (file contains 2 functions, both are converted)
python3 src/cli/main.py -i examples/vision/image_preprocessing.py --test-input test.jpg

# Two functions converted:
# 1. preprocess_image -> .build/output/preprocess_image/
# 2. preprocess_with_color_conversion -> .build/output/preprocess_with_color_conversion/

# Build and run
cd .build/output/preprocess_image
mkdir build && cd build
cmake .. && make
./preprocess_image ../../../../test.jpg
# Validation: ~81% match (bilinear interpolation differences)
```

## Pipeline Mode

Use `# @inference` marker to split Pre/Inf/Post:

```bash
python3 src/cli/main.py -i examples/vision/mnist_inference.py --pipeline --test-input test.jpg
```

Generates: `preprocess.h`, `inference.h` (stub), `postprocess.h`, `main.cpp`

## CLI Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Python source file |
| `-f, --function` | Function name to convert (optional) |
| `--pipeline` | Split Pre/Inf/Post mode |
| `--test-input` | Test input for validation |
| `--no-validate` | Skip validation |
| `-v, --verbose` | Verbose output |

## Documentation

- **STRUCTURE.md** - Architecture and code organization
- **tasks.md** - Development roadmap
- **CLAUDE.md** - Coding standards

## License

MIT License
