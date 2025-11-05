# Quick Start Guide

Get up and running in 5 minutes!

## Installation (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Build C++ extensions
python setup.py build_ext --inplace

# Install package
pip install -e .
```

## Run Demo (3 minutes)

### Option 1: Interactive Demo Script

```bash
python scripts/demo.py
```

This will:
1. ✅ Generate 20 synthetic tracks
2. ✅ Train Random Forest and XGBoost models
3. ✅ Run inference and assign tags
4. ✅ Display results
5. ✅ Save to `./data/demo/results.csv`

### Option 2: GUI Walkthrough

**Step 1: Generate data**
```bash
python -m simulator.main --num-tracks 5 --duration 30 --output-dir ./data/quick
```

**Step 2: Train a model**
```bash
python scripts/train_models.py --data ./data/quick/simulated_tracks.csv --models rf
```

**Step 3: Launch GUI**
```bash
python -m gui.main
```

**Step 4: In the GUI**
1. Open `./data/quick/simulated_tracks.csv`
2. Go to "Model Manager" → Load Random Forest → Select `models/saved/random_forest`
3. Go to "Track Analysis" → Click "Run Inference"
4. View tagged tracks!

## What's Next?

- 📖 Read the [User Guide](docs/USER_GUIDE.md) for detailed usage
- 🔧 Check the [API Reference](docs/API.md) for programming
- 🧪 Run tests: `pytest tests/ -v`
- 🎨 Customize models in `ml/trainer.py`

## Quick Commands

```bash
# Run GUI
python -m gui.main

# Generate synthetic data
python -m simulator.main --num-tracks 10 --duration 60 --format both

# Train models
python scripts/train_models.py --num-tracks 100 --models rf xgb

# Run tests
pytest tests/ -v

# View help
python -m gui.main --help
python -m simulator.main --help
```

## Troubleshooting

**Problem**: C++ build fails
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install
```

**Problem**: GUI won't start
```bash
pip install --upgrade PySide6
```

**Problem**: ImportError
```bash
pip install -e .
```

## System Requirements

- Python 3.8+
- 4GB RAM (8GB recommended)
- 500MB disk space
- C++ compiler (for building extensions)

## Success! 🎉

You should now have:
- ✅ Working installation
- ✅ Synthetic data generated
- ✅ Trained ML models
- ✅ GUI running

For complete documentation, see [README.md](README.md).
