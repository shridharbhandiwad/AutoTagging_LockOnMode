# Quick Start Guide

Get up and running with Airborne Track Tagger in 5 minutes.

## Installation (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Build C++ extensions
python setup.py build_ext --inplace

# Verify installation
python -c "import parsers, ml, gui, simulator; print('✓ All modules loaded')"
```

## Generate and Train (2 minutes)

```bash
# Generate sample data (50 tracks)
python simulator/generate_samples.py --num-tracks 50

# Train models
python examples/train_example.py
```

Expected output:
```
=== Track Behavior Tagger - Training Example ===

Step 1: Generating synthetic training data...
  Generated 30000 measurements for 50 tracks

Step 2: Extracting features...
  Extracted features for 50 tracks

...

Step 6: Saving models...
  Models saved to ./models/

=== Training complete! ===
```

## Run GUI (1 minute)

```bash
python gui/app.py
```

In the GUI:
1. Click "Load Models" → select `./models/` directory
2. Click "Browse Files" → select `./data/samples/scenario1_linear.csv`
3. Click "Parse File"
4. Click "Run Inference"
5. Click any track in the list to see visualization and tags

## Run Inference from Command Line

```bash
python examples/inference_example.py
```

Expected output:
```
=== Track Behavior Tagger - Inference Example ===

Step 1: Loading trained models...
  Loaded 2 models: RandomForest, XGBoost

...

Step 4: Running inference...

  Using RandomForest:
    Track 1:
      Tags: linear_track (0.92)
    Track 2:
      Tags: linear_track (0.88)

=== Inference complete! ===
```

## What's Next?

- Read the [User Guide](docs/USER_GUIDE.md) for detailed usage
- Check [API Reference](docs/API_REFERENCE.md) for programming
- Explore [examples/](examples/) for more code samples
- Run tests: `pytest tests/ -v`

## Troubleshooting

**Problem:** C++ build fails
```bash
# Install build tools
# Ubuntu/Debian:
sudo apt install build-essential python3-dev

# macOS:
xcode-select --install

# Then retry:
pip install pybind11
python setup.py build_ext --inplace
```

**Problem:** GUI won't start
```bash
# Install Qt dependencies
# Ubuntu/Debian:
sudo apt install libxcb-xinerama0

# Then:
pip install --upgrade PySide6
```

**Problem:** "No module named 'cxxlib'"
```bash
# Rebuild C++ extension
python setup.py build_ext --inplace

# Verify:
python -c "import cxxlib; print('✓ C++ module loaded')"
```

## System Requirements

- **OS**: Linux, macOS, Windows 10+
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **RAM**: 4 GB minimum, 8 GB recommended
- **Disk**: 500 MB for installation, 2 GB for data
- **CPU**: Any modern CPU (GPU optional for deep learning)

## Quick Reference

### File Formats Supported
- Binary: `.bin`, `.raw`, `.dump`
- Text: `.csv`, `.txt`, `.json`, `.jsonl`

### Behavior Tags
- `high_speed`: Speed > 400 m/s
- `low_speed`: Speed < 150 m/s
- `high_maneuver`: High acceleration
- `linear_track`: Straight flight
- `climb`: Altitude increasing
- `descent`: Altitude decreasing
- `two_jet`: Large RCS

### Key Commands
```bash
# Generate data
python simulator/generate_samples.py

# Train models
python examples/train_example.py

# Run inference
python examples/inference_example.py

# Start GUI
python gui/app.py

# Run tests
pytest tests/

# Build C++
python setup.py build_ext --inplace
```

## Getting Help

- Documentation: `docs/`
- Examples: `examples/`
- Tests: `tests/` (run to see how components work)
- Issues: File on GitHub

Happy tracking! 🛩️📡
