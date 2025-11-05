# Sample Datasets

This directory contains sample datasets for testing and training the Airborne Track Tagger.

## Generating Samples

To generate sample datasets, run:

```bash
python simulator/generate_samples.py --training --test --num-tracks 100
```

This will create:
- `training_data.csv` - Training dataset with track measurements
- `training_tags.csv` - Ground truth tags for training
- `scenario1_linear.csv` - Test scenario with linear tracks
- `scenario2_highspeed.csv` - Test scenario with high-speed tracks
- `scenario3_maneuvering.csv` - Test scenario with maneuvering tracks
- `scenario4_mixed.csv` - Test scenario with mixed behaviors

## Dataset Format

### Measurement Data (e.g., training_data.csv)

Columns:
- `track_id`: Unique track identifier
- `timestamp`: Time in seconds
- `range`, `azimuth`, `elevation`: Spherical coordinates
- `range_rate`: Radial velocity
- `snr`: Signal-to-noise ratio (dB)
- `rcs`: Radar cross-section (dBsm)
- `doppler`: Doppler frequency shift
- `pos_x`, `pos_y`, `pos_z`: Cartesian position (m)
- `vel_x`, `vel_y`, `vel_z`: Velocity (m/s)
- `acc_x`, `acc_y`, `acc_z`: Acceleration (m/s²)
- `pos_error_x`, `pos_error_y`, `pos_error_z`: Position error estimates
- `vel_error_x`, `vel_error_y`, `vel_error_z`: Velocity error estimates

### Tags Data (e.g., training_tags.csv)

Columns:
- `track_id`: Unique track identifier
- `high_speed`: Boolean, true if mean speed > 400 m/s
- `low_speed`: Boolean, true if mean speed < 150 m/s
- `high_maneuver`: Boolean, true if high acceleration/jerk
- `linear_track`: Boolean, true if low maneuver index
- `climb`: Boolean, true if positive altitude change
- `descent`: Boolean, true if negative altitude change
- `two_jet`: Boolean, true if large RCS (two-engine aircraft)

## Usage

Load sample data:

```python
import pandas as pd

# Load measurements
data = pd.read_csv('data/samples/training_data.csv')

# Load ground truth tags
tags = pd.read_csv('data/samples/training_tags.csv')
```
