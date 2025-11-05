# User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Working with Files](#working-with-files)
5. [Track Analysis](#track-analysis)
6. [ML Models](#ml-models)
7. [Simulator](#simulator)
8. [Export and Reports](#export-and-reports)
9. [FAQ](#faq)

## Introduction

The Airborne Track Behavior Tagging Application automates the analysis of radar track data from airborne trackers operating in lock-on mode. It extracts track behavior patterns using machine learning and provides an intuitive interface for visualization and analysis.

## Installation

### Prerequisites

- Python 3.8 or higher
- C++ compiler (for building extensions)
- 4GB RAM minimum (8GB recommended)
- 500MB disk space

### Step-by-Step Installation

1. **Download and Extract**
   ```bash
   cd /workspace
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Build C++ Extensions**
   ```bash
   python setup.py build_ext --inplace
   ```

5. **Install Package**
   ```bash
   pip install -e .
   ```

6. **Verify Installation**
   ```bash
   track-tagger-gui --help
   pytest tests/ -v
   ```

## Getting Started

### Launching the Application

```bash
track-tagger-gui
```

The main window opens with three tabs:
- **Track Analysis**: Load files and view tracks
- **Model Manager**: Load ML models
- **Simulator**: Generate synthetic data

### Quick Example

1. **Generate Sample Data**
   - Go to Simulator tab
   - Set: 5 tracks, 30 seconds, 10 Hz
   - Click "Run Simulation"
   - Load generated data when prompted

2. **Train a Model** (optional, or use pre-trained)
   ```bash
   python scripts/train_models.py --num-tracks 50
   ```

3. **Load Model**
   - Go to Model Manager tab
   - Click "Load Random Forest"
   - Select `models/saved/random_forest/`

4. **Run Inference**
   - Go to Track Analysis tab
   - Click "Run Inference"
   - View tags in track list and detail pane

## Working with Files

### Supported File Formats

**Binary Files** (`.bin`, `.dat`):
- Default C-struct format (TrackRecord)
- Custom structs (configure in code)
- Supports little-endian and big-endian

**Text Files** (`.csv`, `.txt`, `.log`, `.json`):
- CSV with headers
- Tab-separated values
- Whitespace-delimited
- JSON-lines (one JSON object per line)

### Loading Files

**Method 1: Drag and Drop**
- Drag file from file explorer onto main window
- File type detected automatically

**Method 2: File Dialog**
- Click "Open File" button
- Select file in dialog
- Click "Open"

### File Format Requirements

**CSV Format:**
```csv
track_id,timestamp,range,azimuth,elevation,range_rate,snr,rcs
1,0,5000.0,0.5,0.2,-50.0,30.0,10.0
1,1,4950.0,0.51,0.21,-50.0,29.5,10.1
```

**Binary Format:**
- See `parsers/binary_parser.py` for struct definition
- Default: 116 bytes per record
- Fields: timestamp (8), track_id (4), range (4), azimuth (4), elevation (4), etc.

## Track Analysis

### Track List

The left panel shows all tracks with:
- **Track ID**: Unique identifier
- **Points**: Number of measurements
- **Avg Speed**: Mean velocity magnitude
- **Max Height**: Maximum altitude
- **Flight Time**: Duration in seconds
- **Tags**: Assigned behavior tags

**Sorting**: Click column headers to sort

**Selection**: Click row to view details

### Track Detail View

The right panel shows detailed information:

**Summary Statistics:**
- Flight time, speeds (max/min/mean)
- Heights (max/min)
- Ranges (max/min)
- Maneuver index
- Signal stats (SNR, RCS, Doppler)

**Behavior Tags:**
- Tag names with confidence scores
- Color-coded by confidence

**Time-Series Plots:**
- Range vs Time
- Velocity vs Time
- Height vs Time

**Interactions:**
- Zoom: Mouse wheel
- Pan: Click and drag
- Reset: Right-click menu

## ML Models

### Loading Models

1. Go to **Model Manager** tab
2. Click appropriate button:
   - "Load Random Forest"
   - "Load XGBoost"
   - "Load LSTM"
3. Select model directory
4. Model appears in "Loaded Models" table

### Training Models

**Command Line:**
```bash
# Train on synthetic data
python scripts/train_models.py --num-tracks 100 --models rf xgb

# Train on existing data
python scripts/train_models.py --data tracks.csv --models rf xgb
```

**Output:**
- Trained models in `models/saved/`
- Metrics in `metrics.json`
- Feature importance plots (PNG)

### Running Inference

1. Ensure at least one model is loaded
2. Go to **Track Analysis** tab
3. Click **Run Inference**
4. Progress bar shows status
5. Tags appear in track list and detail view

### Understanding Tags

**Speed Tags:**
- `high_speed`: Mean speed > 300 m/s (Mach 0.88+)
- `low_speed`: Mean speed < 100 m/s

**Maneuver Tags:**
- `high_maneuver`: High acceleration variance
- `linear_track`: Nearly constant velocity

**Altitude Tags:**
- `climb`: Altitude increasing > 100m
- `descent`: Altitude decreasing > 100m
- `hover_like`: Low speed + stable altitude

**Engine Tags:**
- `two_jet`: RCS pattern suggests twin-engine
- `multiengine`: Large RCS (4+ engines)
- `unknown_engine`: Cannot determine

**Confidence Scores:**
- 0.0 - 0.3: Low confidence
- 0.3 - 0.7: Medium confidence
- 0.7 - 1.0: High confidence

## Simulator

### Basic Usage

1. Go to **Simulator** tab
2. Configure scenario:
   - **Number of Tracks**: 1-100
   - **Duration**: 1-600 seconds
   - **Update Rate**: 1-100 Hz
   - **Output Format**: binary/CSV/both
3. Click "Run Simulation"
4. Select output directory
5. Wait for completion

### Scenario Configuration

**Track Types** (probabilistic):
- High-speed (30%): 300-500 m/s
- Low-speed (20%): 50-100 m/s
- High-maneuver (25%): Random accelerations
- Linear (40%): Constant velocity

**Noise Parameters:**
- Position noise: σ = 5m
- Velocity noise: σ = 1 m/s
- Measurement dropout: 5%
- False alarm rate: 2%

### Output Files

After simulation:
- `simulated_tracks.bin` - Binary format
- `simulated_tracks.csv` - CSV format
- `ground_truth_labels.json` - True labels

**Loading Simulated Data:**
- Prompt appears after simulation
- Click "Yes" to load automatically
- Or manually open file later

## Export and Reports

### Exporting Results

1. Click **Export Results** button
2. Choose format:
   - CSV: Human-readable
   - Parquet: Compressed, efficient
3. Select output location
4. File contains:
   - All track data
   - Computed features
   - Assigned tags

### Export File Structure

**CSV Format:**
```csv
track_id,timestamp,range,azimuth,elevation,...,high_speed,low_speed,...
1,0,5000.0,0.5,0.2,...,0.85,0.12,...
```

**Columns:**
- Raw measurements
- Processed data (positions, velocities)
- Signal characteristics
- Tag confidence scores

### Programmatic Export

```python
from feature_store import FeatureStore

store = FeatureStore('./data/feature_store')
store.export_all_tracks('output.csv', format='csv')
```

## FAQ

### Q: What file size limits exist?

**A:** No hard limits, but performance considerations:
- < 1GB: Fast loading
- 1-5GB: Moderate loading time
- \> 5GB: Consider splitting files

### Q: Can I use my own C++ libraries?

**A:** Yes! See Developer Guide for integration steps.

### Q: How accurate are the ML models?

**A:** On synthetic data:
- Random Forest: 85-90% accuracy
- XGBoost: 87-92% accuracy
- LSTM: 88-93% accuracy

Real-world accuracy depends on training data quality.

### Q: Can I add custom tags?

**A:** Yes:
1. Edit `ml/trainer.py` - add to `TAG_DEFINITIONS`
2. Implement labeling logic in `_generate_labels()`
3. Retrain models
4. Update GUI if needed

### Q: What if file format is not recognized?

**A:** Create custom parser:
```python
from parsers.text_parser import TextParser

parser = TextParser()
df = parser.parse_file('myfile.dat', format='whitespace')
```

### Q: How do I visualize multiple tracks together?

**A:** Currently shows one track at a time. For batch visualization, export to CSV and use external tools (matplotlib, Excel, etc.).

### Q: Can I run headless (no GUI)?

**A:** Yes, use Python API:
```python
from parsers import FileDetector
from ml.inference import ModelInference

parser = FileDetector.get_parser('data.bin')
df = parser.parse_to_dataframe('data.bin')

# ... process tracks ...

inference = ModelInference()
# ... load models and run inference ...
```

### Q: What operating systems are supported?

**A:** 
- Linux: Fully supported
- Windows: Supported (tested on Windows 10/11)
- macOS: Supported (tested on macOS 11+)

### Q: How do I report bugs?

**A:** Open a GitHub issue with:
- Python version (`python --version`)
- Operating system
- Error message/traceback
- Steps to reproduce

---

For more details, see:
- API Reference: `docs/API.md`
- Developer Guide: `docs/DEVELOPER.md`
- README: `README.md`
