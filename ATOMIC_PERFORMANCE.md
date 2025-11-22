# Atomic Function Performance Test Results

**Generated:** 2025-11-22 23:21:41

## Summary

- **Total Functions Tested:** 25
- **Successful Builds:** 12
- **Build Failures:** 13
- **Average Accuracy:** 53.09%

## Categories

### NumPy

- **Total:** 11
- **Success:** 3
- **Failed:** 8
- **Avg Accuracy:** 50.12%

| Function | Python (ms) | C++ (ms) | Speedup | Accuracy | Status |
|----------|-------------|----------|---------|----------|--------|
| `test_divide_scalar` | 11.856 ms | 67.344 ms | 0.18x | 92.96% | ✅ PASS |
| `test_reshape` | 8.962 ms | 57.782 ms | 0.16x | 57.40% | ⚠️ LOW_ACCURACY |
| `test_mean` | 14.386 ms | 59.570 ms | 0.24x | 0.00% | ⚠️ LOW_ACCURACY |

**Build Failures:**

- `test_astype_float32`: BUILD_FAILED
- `test_astype_uint8`: BUILD_FAILED
- `test_multiply_scalar`: BUILD_FAILED
- `test_transpose`: BUILD_FAILED
- `test_std`: BUILD_FAILED
- `test_clip`: BUILD_FAILED
- `test_subtract`: BUILD_FAILED
- `test_add`: BUILD_FAILED

### OpenCV

- **Total:** 8
- **Success:** 7
- **Failed:** 1
- **Avg Accuracy:** 69.54%

| Function | Python (ms) | C++ (ms) | Speedup | Accuracy | Status |
|----------|-------------|----------|---------|----------|--------|
| `test_imread` | 10.463 ms | 68.256 ms | 0.15x | 92.96% | ✅ PASS |
| `test_imread_grayscale` | 10.862 ms | 78.994 ms | 0.14x | 46.30% | ⚠️ LOW_ACCURACY |
| `test_resize` | 10.558 ms | 60.339 ms | 0.17x | 82.80% | ✅ PASS |
| `test_cvtColor` | 10.115 ms | 69.354 ms | 0.15x | 92.96% | ✅ PASS |
| `test_cvtColor_gray` | 9.990 ms | 61.952 ms | 0.16x | 46.64% | ⚠️ LOW_ACCURACY |
| `test_GaussianBlur` | 10.016 ms | 75.142 ms | 0.13x | 50.33% | ⚠️ LOW_ACCURACY |
| `test_Canny` | 11.561 ms | 66.773 ms | 0.17x | 74.77% | ⚠️ LOW_ACCURACY |

**Build Failures:**

- `test_threshold`: BUILD_FAILED

### Type Conversion

- **Total:** 6
- **Success:** 2
- **Failed:** 4
- **Avg Accuracy:** 0.00%

| Function | Python (ms) | C++ (ms) | Speedup | Accuracy | Status |
|----------|-------------|----------|---------|----------|--------|
| `test_max_value` | 10.040 ms | 59.519 ms | 0.17x | 0.00% | ⚠️ LOW_ACCURACY |
| `test_min_value` | 10.049 ms | 59.155 ms | 0.17x | 0.00% | ⚠️ LOW_ACCURACY |

**Build Failures:**

- `test_int_conversion`: BUILD_FAILED
- `test_float_conversion`: BUILD_FAILED
- `test_bool_operation`: BUILD_FAILED
- `test_tuple_return`: BUILD_FAILED

## Analysis

### Best Performing Functions

| Function | Accuracy | Category |
|----------|----------|----------|
| `test_imread` | 92.96% | OpenCV |
| `test_cvtColor` | 92.96% | OpenCV |
| `test_divide_scalar` | 92.96% | NumPy |
| `test_resize` | 82.80% | OpenCV |
| `test_Canny` | 74.77% | OpenCV |
| `test_reshape` | 57.40% | NumPy |
| `test_GaussianBlur` | 50.33% | OpenCV |
| `test_cvtColor_gray` | 46.64% | OpenCV |
| `test_imread_grayscale` | 46.30% | OpenCV |
| `test_mean` | 0.00% | NumPy |

### Failed Functions

Functions that failed to build or execute:

- `test_threshold` (OpenCV): BUILD_FAILED
- `test_astype_float32` (NumPy): BUILD_FAILED
- `test_astype_uint8` (NumPy): BUILD_FAILED
- `test_multiply_scalar` (NumPy): BUILD_FAILED
- `test_transpose` (NumPy): BUILD_FAILED
- `test_std` (NumPy): BUILD_FAILED
- `test_clip` (NumPy): BUILD_FAILED
- `test_subtract` (NumPy): BUILD_FAILED
- `test_add` (NumPy): BUILD_FAILED
- `test_int_conversion` (Type Conversion): BUILD_FAILED
- `test_float_conversion` (Type Conversion): BUILD_FAILED
- `test_bool_operation` (Type Conversion): BUILD_FAILED
- `test_tuple_return` (Type Conversion): BUILD_FAILED

## Test Environment

- **Platform:** Linux (WSL2)
- **Compiler:** g++ (C++17)
- **Python:** 3.10+
- **OpenCV:** 4.x
- **Test Image:** config/data/test_image.jpg

## Notes

- Performance measured as average over 10 iterations
- Accuracy measured as percentage of matching array elements
- Speedup = Python time / C++ time
- Status: PASS if accuracy ≥ 75% and build succeeds
- These are individual atomic functions, not composite operations
