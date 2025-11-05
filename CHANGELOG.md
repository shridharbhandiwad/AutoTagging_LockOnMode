# Changelog

All notable changes to the Airborne Track Behavior Tagging Application will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-05

### Added

#### Core Features
- Binary file parser with C-style struct support (little/big endian, packed structs)
- Text file parser (CSV, JSON, whitespace-delimited) with auto-detection
- Automatic file type detection and routing
- Cross-platform GUI application using PySide6
- Drag-and-drop file loading
- Track visualization with time-series plots

#### C++ Integration
- Kalman filter implementation (3D position and velocity tracking)
- Gating and data association utilities
- Python bindings via pybind11
- Batch processing support

#### Feature Store
- TrackFeatures class for comprehensive track data
- Persistence to Parquet, CSV, and JSON formats
- Automatic aggregate feature computation
- Export functionality for all tracks

#### Machine Learning
- Random Forest classifier for behavior tagging
- XGBoost classifier
- LSTM sequence model
- Multi-model ensemble support
- Training pipeline with cross-validation
- Inference pipeline with batch processing
- SHAP-based explainability
- Feature importance visualization

#### Behavior Tags
- Speed tags: high_speed, low_speed
- Maneuver tags: high_maneuver, linear_track
- Altitude tags: climb, descent, hover_like
- Engine type tags: two_jet, multiengine, unknown_engine
- Numeric summaries: flight time, speeds, heights, ranges, maneuver index, SNR/RCS/Doppler means

#### Simulator
- Synthetic track data generation
- Configurable scenarios (track types, noise, dropouts)
- Binary and CSV output formats
- Ground truth label generation
- Real-time and batch modes

#### Testing
- Comprehensive unit tests for all modules
- Integration tests
- Test coverage reporting
- CI/CD pipeline (GitHub Actions)

#### Documentation
- Complete README with quick start
- Detailed User Guide
- Comprehensive API Reference
- Quick Start guide
- Demo scripts
- Example workflows

### Technical Details
- Python 3.8+ support
- Cross-platform (Linux, Windows, macOS)
- Performance: <200ms inference per track
- Binary parsing: ~500k records/second
- Memory efficient with Parquet storage

## [Unreleased]

### Planned Features
- Real-time data streaming support
- Additional ML models (Transformer, GNN)
- Advanced visualization (3D track plots)
- Database backend option
- REST API for remote inference
- Web-based interface
- Docker containerization
- Distributed training support

### Known Issues
- LSTM model requires sequence preparation
- Large files (>5GB) may have slow loading
- Some platforms require additional Qt dependencies for GUI

---

For migration guides and upgrade instructions, see the documentation.
