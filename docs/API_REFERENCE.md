# API Reference

## Parsers Module

### Binary Parser

```python
from parsers.binary.struct_parser import StructParser, FieldDef, Endianness

# Define struct
struct_def = [
    FieldDef("track_id", "uint32_t"),
    FieldDef("timestamp", "uint64_t"),
    FieldDef("position", "float", 3),
]

# Create parser
parser = StructParser(struct_def, Endianness.LITTLE, packed=True)

# Parse file
records = parser.parse_file("data.bin")
```

### Text Parser

```python
from parsers.text.text_parser import TextParser

# Auto-detect and parse
df = TextParser.parse_file("data.csv")

# Specific format
df = TextParser.parse_csv("data.csv", delimiter=',')
df = TextParser.parse_whitespace("data.txt")
df = TextParser.parse_jsonlines("data.jsonl")
```

### File Router

```python
from parsers import FileRouter

# Auto-detect and route
df = FileRouter.parse_file("unknown_file.dat")

# Validate
is_valid = FileRouter.validate_track_data(df)
```

## C++ Wrapper Module

### Kalman Filter

```python
from cxx_wrapper import KalmanFilter

# Create filter
kf = KalmanFilter(
    initial_state=[0, 0, 0, 10, 10, 0],
    process_noise=1.0,
    meas_noise=1.0
)

# Predict
kf.predict(dt=0.1)

# Update
result = kf.update(measurement=[100, 200, 300])
print(result.state)
print(result.residual)
print(result.innovation_magnitude)
```

### Gating

```python
from cxx_wrapper import Gating

# Mahalanobis distance
distance = Gating.mahalanobis_distance(
    measurement=[100, 200, 300],
    prediction=[102, 198, 301],
    covariance=cov_matrix
)

# Gate test
passes = Gating.passes_gate(
    measurement, prediction, covariance,
    gate_threshold=9.21  # Chi-square 99% for 3 DOF
)

# Find best association
best_idx = Gating.find_best_association(
    measurements=[[100,200,300], [105,205,305]],
    prediction=[102,198,301],
    covariance=cov_matrix
)
```

### Feature Extraction

```python
from cxx_wrapper import FeatureExtractor

# Extract features
features = FeatureExtractor.extract_features(
    positions=[[x1,y1,z1], [x2,y2,z2], ...],
    velocities=[[vx1,vy1,vz1], ...],
    accelerations=[[ax1,ay1,az1], ...],
    snr_values=[20, 21, 19, ...],
    rcs_values=[10, 11, 10, ...],
    timestamps=[0, 0.1, 0.2, ...]
)

print(features.max_speed)
print(features.maneuver_index)
print(features.flight_time)
```

## ML Module

### Feature Store

```python
from ml import FeatureStore, TrackFeatures, TrackTags

# Create store
store = FeatureStore('./data/feature_store')

# Store features
features = [
    TrackFeatures(
        track_id=1,
        max_speed=500,
        mean_speed=300,
        # ... more fields
    )
]
store.store_features(features)

# Load features
features_df = store.load_features()

# Export to CSV
store.export_to_csv('./export')
```

### Classical Models

```python
from ml.models.classical_models import ClassicalTrackTagger

# Create tagger
tagger = ClassicalTrackTagger(model_name="RandomForest")

# Train
metrics = tagger.train(features_df, tags_df, test_size=0.2)

# Predict
predictions = tagger.predict(new_features_df)

# Feature importance
importance = tagger.get_feature_importance('high_speed', top_n=10)

# Save/Load
tagger.save('./models/rf')
tagger.load('./models/rf')
```

### Deep Learning Models

```python
from ml.models.deep_models import DeepTrackTagger

# Create tagger
tagger = DeepTrackTagger(model_type="LSTM")

# Prepare sequences
sequences, track_ids = tagger.prepare_sequences(
    measurements_df,
    feature_columns=['range', 'azimuth', 'elevation']
)

# Train
history = tagger.train(
    sequences, labels,
    epochs=50,
    batch_size=32
)

# Predict
predictions = tagger.predict(new_sequences)

# Save/Load
tagger.save('./models/lstm')
tagger.load('./models/lstm')
```

### Training Pipeline

```python
from ml import TrainingPipeline, FeatureStore

store = FeatureStore('./data/feature_store')
pipeline = TrainingPipeline(store)

# Train classical models
results = pipeline.train_classical_models(
    features_df, tags_df,
    model_types=['RandomForest', 'XGBoost', 'LightGBM'],
    test_size=0.2,
    cv_folds=5
)

# Train deep models
results = pipeline.train_deep_models(
    measurements_df, tags_df,
    model_types=['LSTM', 'Transformer'],
    epochs=50
)

# Save all models
pipeline.save_models('./models')

# Compare models
comparison = pipeline.compare_models()
```

### Inference Pipeline

```python
from ml import InferencePipeline

# Create pipeline
pipeline = InferencePipeline()

# Load models
pipeline.load_models('./models', model_names=['RandomForest', 'XGBoost'])

# Predict with specific model
predictions = pipeline.predict(features_df, model_name='RandomForest')

# Ensemble prediction
predictions = pipeline.predict(features_df, ensemble=True)
```

### Explainability

```python
from ml.inference.explainability import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(trained_model)

# Feature importance
importance = analyzer.compute_feature_importance('high_speed', top_n=10)

# SHAP values
shap_values = analyzer.compute_shap_values(features_df, 'high_maneuver')

# Explain specific prediction
explanation = analyzer.explain_prediction(
    track_id=42,
    features_df=features_df,
    tag='high_maneuver',
    top_n=5
)

# Plot feature importance
fig = analyzer.plot_feature_importance('high_speed', save_path='imp.png')

# Plot SHAP summary
fig = analyzer.plot_shap_summary(features_df, 'high_maneuver')

# Generate report
report = analyzer.generate_explanation_report(
    track_id=42,
    features_df=features_df,
    predictions_df=predictions_df
)
```

## Simulator Module

### Track Simulator

```python
from simulator import TrackSimulator, SimulationConfig

# Configure simulation
config = SimulationConfig(
    duration=60.0,
    dt=0.1,
    noise_level=0.1,
    dropout_rate=0.05,
    initial_height=5000.0,
    base_speed=200.0
)

# Create simulator
simulator = TrackSimulator(config)

# Generate single track
track_df = simulator.generate_track('high_speed', track_id=1)

# Generate scenario
scenario_df = simulator.generate_scenario(
    num_tracks=50,
    behavior_distribution={
        'linear': 0.3,
        'high_speed': 0.2,
        'maneuvering': 0.3,
        'climb': 0.2
    }
)

# Generate ground truth
tags_df = simulator.generate_ground_truth_tags(scenario_df)

# Save
simulator.save_scenario(scenario_df, './data/scenario1', format='csv')
```

## GUI Module

### Main Window

```python
from PySide6.QtWidgets import QApplication
from gui import MainWindow

app = QApplication([])
window = MainWindow()
window.show()
app.exec()
```

### Custom Widgets

```python
from gui.widgets import TrackPlotWidget, ModelMetricsWidget

# Track plot widget
plot_widget = TrackPlotWidget()
plot_widget.plot_track(track_data_df)

# Model metrics widget
metrics_widget = ModelMetricsWidget()
metrics_widget.set_models(loaded_models)
metrics_widget.display_metrics(metrics_df)
metrics_widget.plot_feature_importance(importance_dict)
```

### Workers

```python
from gui.workers import FileProcessWorker, InferenceWorker

# File processing worker
worker = FileProcessWorker(file_path)
worker.finished.connect(on_parse_complete)
worker.error.connect(on_parse_error)
worker.start()

# Inference worker
worker = InferenceWorker(inference_pipeline, features_df)
worker.finished.connect(on_inference_complete)
worker.start()
```

## Data Structures

### TrackMeasurement

```python
from ml.feature_store import TrackMeasurement

measurement = TrackMeasurement(
    track_id=1,
    timestamp=100.0,
    range=5000.0,
    azimuth=0.5,
    elevation=0.2,
    range_rate=-100.0,
    snr=20.0,
    rcs=10.0,
    doppler=-500.0
)
```

### ProcessedTrackState

```python
from ml.feature_store import ProcessedTrackState

state = ProcessedTrackState(
    track_id=1,
    timestamp=100.0,
    pos_x=1000.0, pos_y=2000.0, pos_z=3000.0,
    vel_x=-50.0, vel_y=-30.0, vel_z=10.0,
    # ... more fields
)
```

### TrackFeatures

```python
from ml.feature_store import TrackFeatures

features = TrackFeatures(
    track_id=1,
    max_speed=500.0,
    min_speed=100.0,
    mean_speed=300.0,
    std_speed=50.0,
    # ... more fields
)
```

### TrackTags

```python
from ml.feature_store import TrackTags

tags = TrackTags(
    track_id=1,
    high_speed=True,
    low_speed=False,
    high_maneuver=False,
    linear_track=True,
    high_speed_conf=0.95,
    model_name='RandomForest',
    inference_time_ms=150.0
)
```

## Error Handling

All modules raise standard Python exceptions:

```python
try:
    data = FileRouter.parse_file('data.bin')
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Parse error: {e}")
```

## Logging

Enable logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Or use loguru
from loguru import logger
logger.info("Processing track {}", track_id)
```
