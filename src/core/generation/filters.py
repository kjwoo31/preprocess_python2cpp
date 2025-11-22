"""Custom Jinja2 filters for C++ code generation."""


def cpp_type(python_type: str) -> str:
    """Convert Python type hint to C++ type."""
    type_map = {
        "str": "std::string",
        "string": "std::string",
        "int": "int",
        "float": "float",
        "bool": "bool",
        "numpy.ndarray": "cv::Mat",
        "np.ndarray": "cv::Mat",
        "PIL.Image.Image": "cv::Mat",
        "tuple[int, int]": "std::pair<int, int>",
    }

    if "ndarray" in python_type:
        return "cv::Mat"
    elif "tuple" in python_type:
        return "std::tuple<...>"
    elif "list" in python_type:
        return "std::vector<...>"

    return type_map.get(python_type, "auto")


def arithmetic_op(operator: str) -> str:
    """Convert operator name to C++ operator symbol."""
    op_map = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
        "floor_divide": "/",
        "modulo": "%",
        "power": "pow",
    }
    return op_map.get(operator, operator)


def format_args(args: list) -> str:
    """Format argument list for C++ function call."""
    return ", ".join(str(arg) for arg in args)


def cpp_lib(python_lib: str) -> str:
    """Convert Python library name to C++ library/namespace name."""
    lib_map = {
        "cv2": "cv",
        "numpy": "np",
        "np": "np",
        "librosa": "librosa",
        "PIL": "cv",
    }
    return lib_map.get(python_lib, python_lib)
