# API Reference

Complete API documentation for the Airborne Track Behavior Tagging Application.

## Table of Contents

1. [Parsers](#parsers)
2. [Feature Store](#feature-store)
3. [ML Models](#ml-models)
4. [C++ Library](#c-library)
5. [Simulator](#simulator)

---

## Parsers

### BinaryParser

Parse binary files using C-style struct definitions.

```python
from parsers.binary_parser import BinaryParser, StructDefinition, FieldDefinition

# Create struct definition
fields = [
    FieldDefinition('timestamp', 'uint64_t'),
    FieldDefinition('value', 'float', count=3)  # Array
]
struct_def = StructDefinition('MyStruct', fields, packed=True)

# Create parser
parser = BinaryParser(struct_def)

# Parse file
records = parser.parse_file('data.bin')  # List[Dict]
df = parser.parse_to_dataframe('data.bin')  # pandas.DataFrame
```

#### StructDefinition

**Constructor:**
```python
StructDefinition(
    name: str,
    fields: List[FieldDefinition],
    packed: bool = False,
    endianness: Endianness = Endianness.LITTLE
)
```

**Methods:**
- `get_format_string() -> str`: Returns struct format string
- `get_size() -> int`: Returns total size in bytes

#### FieldDefinition

**Constructor:**
```python
FieldDefinition(
    name: str,
    type: str,  # 'uint8_t', 'int16_t', 'float', 'double', etc.
    count: int = 1  # For arrays
)
```

**Supported Types:**
- `uint8_t`, `int8_t`
- `uint16_t`, `int16_t`
- `uint32_t`, `int32_t`
- `uint64_t`, `int64_t`
- `float`, `double`
- `char`

### TextParser

Parse text-based files (CSV, JSON, whitespace-delimited).

```python
from parsers.text_parser import TextParser

parser = TextParser()

# Auto-detect format
df = parser.parse_file('data.csv')

# Specify format
df = parser.parse_file('data.txt', format='whitespace')

# Parse and group by track
tracks = parser.parse_to_tracks('data.csv')  # Dict[int, DataFrame]
```

**Methods:**
- `detect_format(filepath: str) -> str`: Auto-detect file format
- `parse_file(filepath: str, format: Optional[str]) -> DataFrame`
- `parse_to_tracks(filepath: str, format: Optional[str]) -> Dict[int, DataFrame]`
- `standardize_columns(df: DataFrame) -> DataFrame`: Normalize column names

### FileDetector

Automatic file type detection.

```python
from parsers.file_detector import FileDetector

# Detect file type
file_type, subtype = FileDetector.detect_file_type('data.bin')
# Returns: ('binary', None) or ('text', 'csv')

# Get appropriate parser
parser = FileDetector.get_parser('data.bin')
```

---

## Feature Store

### TrackFeatures

Represents features for a single track.

```python
from feature_store import TrackFeatures

track = TrackFeatures(
    track_id=1,
    timestamps=[0, 1, 2],
    ranges=[5000, 4950, 4900],
    azimuths=[0.5, 0.52, 0.54],
    elevations=[0.2, 0.21, 0.22],
    range_rates=[-50, -50, -50],
    positions=[[100, 200, 5000], ...],
    velocities=[[50, 50, 10], ...],
    accelerations=[[0, 0, 0], ...],
    kalman_states=[],
    kalman_covariances=[],
    innovations=[],
    snr_values=[30, 30, 30],
    rcs_values=[10, 10, 10],
    doppler_values=[1000, 1000, 1000],
    pos_errors=[],
    vel_errors=[],
)
```

**Properties:**
- `track_id: int`
- `timestamps: List[float]`
- `ranges, azimuths, elevations, range_rates: List[float]`
- `positions, velocities, accelerations: List[List[float]]`
- `kalman_states, kalman_covariances, innovations: List[List[float]]`
- `snr_values, rcs_values, doppler_values: List[float]`
- `pos_errors, vel_errors: List[List[float]]`
- `flight_time, max_speed, min_speed, mean_speed, std_speed: float`
- `max_height, min_height, max_range, min_range: float`
- `maneuver_index, snr_mean, rcs_mean, doppler_mean: float`
- `tags: Dict[str, float]`

**Methods:**
- `compute_aggregate_features()`: Compute derived features
- `to_dict() -> Dict`: Convert to dictionary
- `to_dataframe() -> DataFrame`: Convert to pandas DataFrame
- `get_feature_vector() -> np.ndarray`: Get ML feature vector (13 features)

### FeatureStore

Persist and retrieve track features.

```python
from feature_store import FeatureStore

store = FeatureStore('./data/feature_store')

# Save track
store.save_track(track, format='parquet')  # or 'csv', 'json'

# Load track
track = store.load_track(track_id=1, format='parquet')

# Export all tracks
store.export_all_tracks('output.csv', format='csv')

# Get all track IDs
track_ids = store.get_all_track_ids()
```

---

## ML Models

### Base Model Interface

All models inherit from `BaseModel`:

```python
class BaseModel:
    def fit(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Train model."""
        
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict labels."""
        
    def save(self, path: str):
        """Save model to disk."""
        
    def load(self, path: str):
        """Load model from disk."""
```

### RandomForestTagger

```python
from ml.models import RandomForestTagger

model = RandomForestTagger(
    n_estimators=100,
    max_depth=10
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
# Returns: {'high_speed': array([0.85, 0.12, ...]), ...}

# Feature importance
importances = model.get_feature_importance()
# Returns: {'high_speed': array([0.15, 0.08, ...]), ...}

# Save/Load
model.save('./models/saved/rf')
model.load('./models/saved/rf')
```

### XGBoostTagger

```python
from ml.models import XGBoostTagger

model = XGBoostTagger(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

# Same API as RandomForestTagger
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### LSTMTagger

```python
from ml.models import LSTMTagger

model = LSTMTagger(
    input_dim=13,
    hidden_dim=64,
    num_layers=2,
    num_tags=10
)

# Train on sequences
model.fit(
    X_seq,  # Shape: (n_samples, seq_len, n_features)
    y,
    epochs=50,
    batch_size=32,
    lr=0.001
)

# Predict
predictions = model.predict(X_seq)
```

### ModelTrainer

```python
from ml.trainer import ModelTrainer

trainer = ModelTrainer()

# Prepare data from tracks
X, y = trainer.prepare_data(tracks)

# Train model
model = RandomForestTagger()
metrics = trainer.train_model(
    model,
    tracks,
    test_size=0.2,
    random_state=42
)

# Cross-validation
cv_metrics = trainer.cross_validate(model, tracks, n_folds=5)

# Save metrics
trainer.save_metrics(metrics, 'metrics.json')
```

### ModelInference

```python
from ml.inference import ModelInference

inference = ModelInference()

# Add models
inference.add_model('RF', rf_model, weight=1.0)
inference.add_model('XGB', xgb_model, weight=1.5)

# Single track prediction
results = inference.predict_single_track(track, use_ensemble=True)
# Returns: {
#     'RF': {'tags': {...}, 'inference_time_ms': 15},
#     'XGB': {'tags': {...}, 'inference_time_ms': 12},
#     'ensemble': {'tags': {...}, 'inference_time_ms': 27}
# }

# Batch prediction
tags = inference.predict_batch(tracks, model_name='RF')

# Apply tags to track
track = inference.apply_tags_to_track(track, model_name='ensemble')

# Model comparison
comparison = inference.get_model_comparison(tracks)
```

### ExplainabilityAnalyzer

```python
from ml.explainability import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(model, feature_names)

# Feature importance
importances = analyzer.get_feature_importance()
# Returns: {'high_speed': {'flight_time': 0.15, 'max_speed': 0.25, ...}}

# SHAP values
shap_results = analyzer.compute_shap_values(X, tag_name='high_speed')

# Plots
analyzer.plot_feature_importance('high_speed', 'importance.png', top_n=10)
analyzer.plot_shap_summary(X, 'high_maneuver', 'shap.png')

# Explain single prediction
explanation = analyzer.explain_prediction(features, 'high_speed')
# Returns: {
#     'prediction': 0.85,
#     'feature_values': {...},
#     'feature_contributions': {...}
# }
```

---

## C++ Library

### KalmanFilter

```python
from cxxlib import KalmanFilter

kf = KalmanFilter()

# Initialize
kf.initialize(
    position=[0.0, 0.0, 1000.0],  # [x, y, z]
    velocity=[100.0, 0.0, 0.0]    # [vx, vy, vz]
)

# Predict
kf.predict(dt=0.1)  # Time step in seconds

# Update with measurement
kf.update(
    measurement=[10.0, 5.0, 1010.0],
    measurement_noise=[1.0, 1.0, 1.0]
)

# Get results
state = kf.get_state()  # [x, y, z, vx, vy, vz]
covariance = kf.get_covariance()  # 6x6 matrix (flattened)
innovation = kf.get_innovation()  # Residual

# Reset
kf.reset()
```

### Batch Kalman Processing

```python
from cxxlib import run_kalman
import numpy as np

# Prepare measurements
measurements = np.array([
    [0.0, 0.0, 1000.0],
    [10.0, 5.0, 1010.0],
    [20.0, 10.0, 1020.0],
])

# Run Kalman filter
result = run_kalman(measurements, dt=0.1)

states = result['states']  # List of state vectors
innovations = result['innovations']  # List of residuals
final_cov = result['final_covariance']  # Final covariance matrix
```

### Gating

```python
from cxxlib import Gating

# Mahalanobis distance
distance = Gating.mahalanobis_distance(
    measurement=[10.0, 5.0, 1010.0],
    predicted_state=[9.0, 4.5, 1009.0, 100.0, 50.0, 10.0],
    covariance=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]  # 3x3 flattened
)

# Check gate
is_valid = Gating.is_within_gate(distance, gate_threshold=9.21)

# Compute cost matrix for association
measurements_flat = [...]  # Flattened measurements
tracks_flat = [...]  # Flattened track states
cost_matrix = Gating.compute_cost_matrix(
    measurements_flat,
    tracks_flat,
    num_measurements=5,
    num_tracks=3
)
```

---

## Simulator

### SimulationScenario

```python
from simulator import SimulationScenario

scenario = SimulationScenario(
    num_tracks=10,
    duration_seconds=60.0,
    update_rate_hz=10.0,
    
    # Track behavior distribution
    prob_high_speed=0.3,
    prob_low_speed=0.2,
    prob_high_maneuver=0.25,
    prob_linear=0.4,
    
    # Noise parameters
    position_noise_std=5.0,
    velocity_noise_std=1.0,
    measurement_dropout_prob=0.05,
    false_alarm_rate=0.02,
    
    # Output
    output_format='both',  # 'binary', 'csv', or 'both'
    real_time_mode=False
)
```

### TrackSimulator

```python
from simulator import TrackSimulator, SimulationScenario

scenario = SimulationScenario(num_tracks=5, duration_seconds=30.0)
simulator = TrackSimulator(scenario)

# Run simulation
output_files = simulator.run_simulation('./data/output')
# Returns: ['./data/output/simulated_tracks.bin', ...]

# Manual step-by-step
simulator.initialize_tracks()
for i in range(100):
    measurements = simulator.step()
    # Process measurements...
```

---

## Constants and Enumerations

### Endianness

```python
from parsers.binary_parser import Endianness

Endianness.LITTLE  # Little-endian (default)
Endianness.BIG     # Big-endian
Endianness.NATIVE  # Native byte order
```

### Tag Definitions

```python
from ml.trainer import ModelTrainer

tags = ModelTrainer.TAG_DEFINITIONS
# {
#     'high_speed': 'Speed > 300 m/s',
#     'low_speed': 'Speed < 100 m/s',
#     'high_maneuver': 'High acceleration variance',
#     ...
# }
```

---

## Error Handling

All APIs use standard Python exceptions:

```python
try:
    parser = BinaryParser(struct_def)
    records = parser.parse_file('data.bin')
except FileNotFoundError:
    print("File not found")
except struct.error:
    print("Invalid binary format")
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

---

For usage examples, see the User Guide and code samples in the repository.
