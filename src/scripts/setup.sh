#!/bin/bash
# Setup development environment

echo "Setting up development environment..."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "✓ Setup complete"
echo "Run 'source venv/bin/activate' to activate the environment"
