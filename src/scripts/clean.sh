#!/bin/bash
# Clean generated files and caches

echo "Cleaning generated files..."
rm -rf .build/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
echo "✓ Clean complete"
