# Contributing

## Getting Started

**Before contributing**, read:
- **CLAUDE.md** - Coding standards and principles
- **tasks.md** - Current priorities
- **STRUCTURE.md** - Code architecture

## Development Setup

```bash
# Install dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Type check
make type-check
```

## Coding Standards

### Python
- Follow PEP 8
- Use type hints (Python 3.10+)
- Functions ≤50 lines
- Write pytest tests

### C++
- Modern C++ (C++17/C++20)
- Google C++ Style Guide
- RAII and smart pointers
- Header-only preferred

### All Code
- **Single Responsibility Principle** (mandatory)
- Self-documenting code
- Minimize comments
- DRY and KISS

## Pull Requests

1. Create feature branch
2. Follow coding standards
3. Add tests
4. Ensure CI passes
5. Reference issue number

## Questions?

See **CLAUDE.md** for detailed guidelines.
