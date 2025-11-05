# Installation Instructions

## System Requirements

### Minimum
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 4 GB
- **Disk**: 1 GB free space
- **CPU**: Any modern CPU (x86-64)

### Recommended
- **RAM**: 8 GB or more
- **GPU**: NVIDIA GPU with CUDA support (for deep learning)
- **Disk**: 5 GB free space (for data and models)

## Installation Steps

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-dev build-essential cmake
sudo apt install -y libxcb-xinerama0  # For GUI
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Python (if needed)
brew install python@3.11
```

**Windows:**
1. Install [Python 3.11](https://www.python.org/downloads/) (check "Add to PATH")
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
   - Select "Desktop development with C++"
3. Install [CMake](https://cmake.org/download/)

### 2. Clone Repository

```bash
git clone <repository-url>
cd workspace
```

### 3. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Build C++ Extensions

**Linux/macOS:**
```bash
python setup.py build_ext --inplace
```

**Windows:**
```bash
python setup.py build_ext --inplace --compiler=msvc
```

### 5. Verify Installation

```bash
# Test imports
python -c "import parsers, ml, gui, simulator; print('✓ All modules loaded')"

# Test C++ extension
python -c "import cxxlib; print('✓ C++ module loaded')"

# Run tests
pytest tests/ -v
```

## Alternative: Using CMake for C++ Build

If you prefer to build C++ separately:

```bash
cd cxx_wrapper
mkdir build && cd build
cmake ..
make
make install
cd ../..
```

## Troubleshooting

### Problem: "pybind11 not found"

```bash
pip install pybind11
```

### Problem: C++ compiler not found

**Linux:**
```bash
sudo apt install build-essential
```

**macOS:**
```bash
xcode-select --install
```

**Windows:**
- Install Visual Studio Build Tools
- Restart terminal after installation

### Problem: "Qt platform plugin" error (GUI)

**Linux:**
```bash
sudo apt install libxcb-xinerama0 libxcb-cursor0
```

**macOS/Windows:**
```bash
pip install --upgrade PySide6
```

### Problem: ImportError for numpy/pandas

```bash
pip install --upgrade numpy pandas scipy
```

### Problem: CUDA/GPU errors (optional, for deep learning)

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Optional: Development Installation

For development with editable install:

```bash
pip install -e .
```

## Verify Installation Checklist

- [ ] Python 3.8+ installed
- [ ] All pip packages installed
- [ ] C++ extension built successfully
- [ ] Tests pass
- [ ] GUI launches without errors
- [ ] Can import all modules

## Post-Installation

### Generate Sample Data

```bash
python simulator/generate_samples.py --num-tracks 50
```

### Train Example Models

```bash
python examples/train_example.py
```

### Launch GUI

```bash
python gui/app.py
```

## Uninstall

```bash
# Remove virtual environment
deactivate
rm -rf venv/

# Remove build artifacts
rm -rf build/ dist/ *.egg-info
rm -rf cxx_wrapper/build/

# Remove data and models (if desired)
rm -rf data/ models/
```

## Docker Installation (Alternative)

A Dockerfile is not provided but can be created:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python setup.py build_ext --inplace

CMD ["python", "gui/app.py"]
```

## Support

For installation issues:
- Check system logs
- Verify Python version: `python --version`
- Check installed packages: `pip list`
- Review error messages carefully
- File an issue on GitHub with full error output
