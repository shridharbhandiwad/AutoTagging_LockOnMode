# User Guide

## Installation

### Prerequisites

- Python 3.8 or higher
- C++ compiler (GCC, Clang, or MSVC)
- CMake 3.15+ (for C++ extension)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Build C++ Extensions

**Linux/Mac:**
```bash
python setup.py build_ext --inplace
```

**Windows:**
```bash
python setup.py build_ext --inplace --compiler=msvc
```

### Step 3: Verify Installation

```bash
python -c "import cxxlib; print('C++ extension loaded successfully')"
pytest tests/ -v
```

## Getting Started

### 1. Generate Sample Data

First, generate synthetic training data:

```bash
python simulator/generate_samples.py --training --test --num-tracks 100
```

This creates:
- `data/samples/training_data.csv` - Track measurements
- `data/samples/training_tags.csv` - Ground truth labels
- `data/samples/scenario*.csv` - Test scenarios

### 2. Train Models

```python
import pandas as pd
from ml import TrainingPipeline, FeatureStore

# Load data
features_df = pd.read_csv('data/samples/training_data.csv')
tags_df = pd.read_csv('data/samples/training_tags.csv')

# Initialize
store = FeatureStore('./data/feature_store')
pipeline = TrainingPipeline(store)

# Train (this may take several minutes)
results = pipeline.train_classical_models(
    features_df, tags_df,
    model_types=['RandomForest', 'XGBoost'],
    test_size=0.2,
    cv_folds=5
)

# Save models
pipeline.save_models('./models')
```

### 3. Run GUI Application

```bash
python gui/app.py
```

The GUI will open. You can now:
1. Drag and drop track data files
2. Click "Parse File" to load data
3. Click "Load Models" to load trained models
4. Click "Run Inference" to tag tracks
5. Select tracks to view details and explanations

## Using the GUI

### Main Window Layout

```
┌─────────────────────────────────────────────────┐
│ [Browse] [Parse] [Load Models] [Run] [Export]  │
├──────────────┬──────────────────────────────────┤
│ File Info    │ Track Visualization              │
│              │                                  │
│ Drag & Drop  │ [Range vs Time]                  │
│ Area         │ [Velocity vs Time]                │
│              │ [Height vs Time]                  │
├──────────────┤ [2D Trajectory]                  │
│ Track List   │                                  │
│              ├──────────────────────────────────┤
│ ID │ #Meas  │ Model Metrics                     │
│  1 │  600   │                                  │
│  2 │  550   │ [Accuracy | Precision | F1]      │
│  3 │  700   │                                  │
│              ├──────────────────────────────────┤
│              │ Tags & Explanations              │
│              │ high_speed: True (0.95)          │
│              │ Top features: mean_speed, ...    │
└──────────────┴──────────────────────────────────┘
```

### Loading Files

**Method 1: Drag and Drop**
- Drag a file from file explorer
- Drop into the gray "Drag & Drop" area
- File type is auto-detected

**Method 2: Browse**
- Click "Browse Files"
- Select file in dialog
- Click "Parse File"

**Supported Formats:**
- Binary: `.bin`, `.raw`, `.dump`
- Text: `.csv`, `.txt`, `.json`, `.jsonl`

### Viewing Tracks

1. After parsing, tracks appear in the Track List
2. Click a row to select a track
3. View time-series plots in right panel:
   - **Range vs Time**: Track distance from radar
   - **Velocity vs Time**: Speed profile
   - **Height vs Time**: Altitude changes
   - **2D Trajectory**: Top-view flight path

### Running Inference

1. Click "Load Models"
2. Select directory containing trained models
3. Models are loaded and shown in metrics tab
4. Click "Run Inference"
5. Tags appear in track list and details panel

### Understanding Tags

Each track gets binary tags:
- **high_speed**: Mean speed > 400 m/s
- **low_speed**: Mean speed < 150 m/s
- **high_maneuver**: High acceleration/jerk
- **linear_track**: Straight-line flight
- **climb**: Positive altitude change
- **descent**: Negative altitude change

Plus confidence scores (0-1) for each tag.

### Viewing Explanations

Select a track with tags to see:
- **Top Contributing Features**: Which features most influenced prediction
- **Feature Values**: Actual values for this track
- **SHAP Values**: Positive/negative contributions

### Exporting Results

1. Click "Export CSV"
2. Choose location
3. Exports all tracks with:
   - Measurements
   - Extracted features
   - Predicted tags
   - Confidence scores

## Command-Line Usage

### Parse Files

```python
from parsers import FileRouter

# Auto-detect and parse
df = FileRouter.parse_file('track_data.bin')
print(df.head())
```

### Extract Features

```python
from cxx_wrapper import FeatureExtractor

# Extract for one track
features = FeatureExtractor.extract_features(
    positions, velocities, accelerations,
    snr_values, rcs_values, timestamps
)

print(f"Max speed: {features.max_speed}")
print(f"Maneuver index: {features.maneuver_index}")
```

### Run Inference

```python
from ml import InferencePipeline
import pandas as pd

# Load models
pipeline = InferencePipeline()
pipeline.load_models('./models')

# Load features
features_df = pd.read_csv('features.csv')

# Predict
predictions = pipeline.predict(features_df)

# Show results
for _, row in predictions.iterrows():
    print(f"Track {row['track_id']}:")
    print(f"  High speed: {row['high_speed']} ({row['high_speed_conf']:.2f})")
    print(f"  High maneuver: {row['high_maneuver']} ({row['high_maneuver_conf']:.2f})")
```

### Explain Predictions

```python
from ml.inference.explainability import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(model)

# Explain one track
explanation = analyzer.explain_prediction(
    track_id=42,
    features_df=features_df,
    tag='high_speed',
    top_n=5
)

print("Top contributing features:")
for feature, value in explanation.items():
    print(f"  {feature}: {value:.3f}")
```

## Advanced Usage

### Custom Track Behaviors

Add new behavior models:

```python
from simulator import TrackBehavior, TrackSimulator

class CustomBehavior(TrackBehavior):
    def generate_trajectory(self):
        # Implement custom logic
        return {
            'time': t,
            'pos_x': x, 'pos_y': y, 'pos_z': z,
            'vel_x': vx, 'vel_y': vy, 'vel_z': vz,
            'acc_x': ax, 'acc_y': ay, 'acc_z': az
        }

# Register and use
TrackSimulator.BEHAVIOR_MAP['custom'] = CustomBehavior
```

### Custom Binary Structures

Define your own struct format:

```python
from parsers.binary.struct_parser import StructParser, FieldDef, Endianness

struct_def = [
    FieldDef("header", "uint32_t"),
    FieldDef("track_id", "uint16_t"),
    FieldDef("measurement_type", "uint8_t"),
    FieldDef("values", "float", 10),  # Array of 10 floats
]

parser = StructParser(struct_def, Endianness.BIG, packed=True)
records = parser.parse_file('custom_format.bin')
```

### Ensemble Models

Combine multiple models:

```python
pipeline = InferencePipeline()
pipeline.load_models('./models', model_names=['RandomForest', 'XGBoost', 'LightGBM'])

# Ensemble prediction (weighted average)
predictions = pipeline.predict(features_df, ensemble=True)
```

### Real-Time Simulation

Generate data in real-time:

```python
import time
from simulator import TrackSimulator

simulator = TrackSimulator()

while True:
    # Generate one time step
    track_df = simulator.generate_track('maneuvering', track_id=1)
    
    # Process immediately
    # ... (run inference, update GUI, etc.)
    
    time.sleep(0.1)  # 100ms update rate
```

## Troubleshooting

### C++ Extension Won't Build

**Error:** "pybind11 not found"
```bash
pip install pybind11
```

**Error:** "Compiler not found"
- Linux: `sudo apt install build-essential`
- Mac: `xcode-select --install`
- Windows: Install Visual Studio Build Tools

### Models Take Too Long to Train

- Reduce `num_tracks` in sample generation
- Use fewer `cv_folds` (try 2 instead of 5)
- Train only RandomForest first (fastest)
- Use smaller `test_size` (try 0.3)

### GUI Won't Start

**Error:** "No module named 'PySide6'"
```bash
pip install PySide6
```

**Error:** "Qt platform plugin could not be initialized"
- Linux: `sudo apt install libxcb-xinerama0`
- Mac: Should work out of box
- Windows: Reinstall PySide6

### Predictions Are All False

- Check that training data has balanced classes
- Verify feature extraction is correct
- Try different models (XGBoost often works better)
- Increase training data size

### File Parse Errors

- Check file is not corrupted
- Verify endianness setting
- For binary: ensure struct definition matches file format
- For text: check delimiter and encoding

## Tips and Best Practices

1. **Always generate test data first** to verify the pipeline
2. **Start with RandomForest** - fastest and most interpretable
3. **Use cross-validation** to catch overfitting early
4. **Check feature importance** to understand models
5. **Export results frequently** to avoid losing work
6. **Use ensemble** for best accuracy on real data
7. **Visualize tracks** before tagging to spot issues

## Next Steps

- Read [API Reference](API_REFERENCE.md) for detailed API
- See [Developer Guide](DEVELOPER_GUIDE.md) to extend the system
- Check `examples/` for code samples
- Run tests to understand components: `pytest tests/ -v`
