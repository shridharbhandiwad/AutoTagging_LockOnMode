#!/bin/bash
# Build script for Airborne Track Tagger

set -e

echo "=== Building Airborne Track Tagger ==="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Build C++ extensions
echo ""
echo "Building C++ extensions..."
python3 setup.py build_ext --inplace

# Run tests
echo ""
echo "Running tests..."
pytest tests/ -v --tb=short

echo ""
echo "=== Build complete! ==="
echo ""
echo "Next steps:"
echo "1. Generate sample data: python3 simulator/generate_samples.py"
echo "2. Train models: python3 examples/train_example.py"
echo "3. Run GUI: python3 gui/app.py"
