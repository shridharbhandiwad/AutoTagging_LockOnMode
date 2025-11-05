"""Feature store implementation using Parquet and CSV."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class TrackFeatures:
    """Features for a single track."""
    track_id: int
    
    # Time-series raw data
    timestamps: List[float]
    ranges: List[float]
    azimuths: List[float]
    elevations: List[float]
    range_rates: List[float]
    
    # Processed data
    positions: List[List[float]]  # [[x,y,z], ...]
    velocities: List[List[float]]
    accelerations: List[List[float]]
    
    # Kalman filter outputs
    kalman_states: List[List[float]]
    kalman_covariances: List[List[float]]
    innovations: List[List[float]]
    
    # Signal characteristics
    snr_values: List[float]
    rcs_values: List[float]
    doppler_values: List[float]
    
    # Errors
    pos_errors: List[List[float]]
    vel_errors: List[List[float]]
    
    # Derived aggregate features
    flight_time: float = 0.0
    max_speed: float = 0.0
    min_speed: float = 0.0
    mean_speed: float = 0.0
    std_speed: float = 0.0
    max_height: float = 0.0
    min_height: float = 0.0
    max_range: float = 0.0
    min_range: float = 0.0
    maneuver_index: float = 0.0
    snr_mean: float = 0.0
    rcs_mean: float = 0.0
    doppler_mean: float = 0.0
    
    # Behavior tags (to be filled by ML)
    tags: Dict[str, float] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        self.compute_aggregate_features()
    
    def compute_aggregate_features(self):
        """Compute aggregate features from time-series data."""
        if len(self.timestamps) < 2:
            return
        
        # Flight time
        self.flight_time = self.timestamps[-1] - self.timestamps[0]
        
        # Speed statistics
        speeds = [np.linalg.norm(v) for v in self.velocities if len(v) == 3]
        if speeds:
            self.max_speed = max(speeds)
            self.min_speed = min(speeds)
            self.mean_speed = np.mean(speeds)
            self.std_speed = np.std(speeds)
        
        # Height statistics (z-coordinate)
        heights = [p[2] for p in self.positions if len(p) == 3]
        if heights:
            self.max_height = max(heights)
            self.min_height = min(heights)
        
        # Range statistics
        if self.ranges:
            self.max_range = max(self.ranges)
            self.min_range = min(self.ranges)
        
        # Maneuver index (based on acceleration variance)
        if len(self.accelerations) > 1:
            accel_mags = [np.linalg.norm(a) for a in self.accelerations if len(a) == 3]
            if accel_mags:
                self.maneuver_index = np.std(accel_mags)
        
        # Signal statistics
        if self.snr_values:
            self.snr_mean = np.mean(self.snr_values)
        if self.rcs_values:
            self.rcs_mean = np.mean(self.rcs_values)
        if self.doppler_values:
            self.doppler_mean = np.mean(self.doppler_values)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert time-series data to DataFrame."""
        data = {
            'track_id': [self.track_id] * len(self.timestamps),
            'timestamp': self.timestamps,
            'range': self.ranges,
            'azimuth': self.azimuths,
            'elevation': self.elevations,
            'range_rate': self.range_rates,
        }
        
        # Add positional data
        if self.positions:
            data['pos_x'] = [p[0] if len(p) > 0 else np.nan for p in self.positions]
            data['pos_y'] = [p[1] if len(p) > 1 else np.nan for p in self.positions]
            data['pos_z'] = [p[2] if len(p) > 2 else np.nan for p in self.positions]
        
        if self.velocities:
            data['vel_x'] = [v[0] if len(v) > 0 else np.nan for v in self.velocities]
            data['vel_y'] = [v[1] if len(v) > 1 else np.nan for v in self.velocities]
            data['vel_z'] = [v[2] if len(v) > 2 else np.nan for v in self.velocities]
        
        if self.snr_values:
            data['snr'] = self.snr_values
        if self.rcs_values:
            data['rcs'] = self.rcs_values
        if self.doppler_values:
            data['doppler'] = self.doppler_values
        
        return pd.DataFrame(data)
    
    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector for ML model (aggregate features only)."""
        features = [
            self.flight_time,
            self.max_speed,
            self.min_speed,
            self.mean_speed,
            self.std_speed,
            self.max_height,
            self.min_height,
            self.max_range,
            self.min_range,
            self.maneuver_index,
            self.snr_mean,
            self.rcs_mean,
            self.doppler_mean,
        ]
        return np.array(features, dtype=np.float32)


class FeatureStore:
    """Store and retrieve track features."""
    
    def __init__(self, base_path: str = "./data/feature_store"):
        """Initialize feature store."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.tracks_cache: Dict[int, TrackFeatures] = {}
    
    def save_track(self, track: TrackFeatures, format: str = 'parquet'):
        """Save track features to storage."""
        track_id = track.track_id
        self.tracks_cache[track_id] = track
        
        if format == 'parquet':
            self._save_parquet(track)
        elif format == 'csv':
            self._save_csv(track)
        elif format == 'json':
            self._save_json(track)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _save_parquet(self, track: TrackFeatures):
        """Save as Parquet file."""
        df = track.to_dataframe()
        filepath = self.base_path / f"track_{track.track_id}.parquet"
        df.to_parquet(filepath, index=False)
        
        # Save metadata separately
        metadata = {
            'track_id': track.track_id,
            'flight_time': track.flight_time,
            'max_speed': track.max_speed,
            'min_speed': track.min_speed,
            'mean_speed': track.mean_speed,
            'std_speed': track.std_speed,
            'max_height': track.max_height,
            'min_height': track.min_height,
            'max_range': track.max_range,
            'min_range': track.min_range,
            'maneuver_index': track.maneuver_index,
            'snr_mean': track.snr_mean,
            'rcs_mean': track.rcs_mean,
            'doppler_mean': track.doppler_mean,
            'tags': track.tags,
        }
        
        meta_filepath = self.base_path / f"track_{track.track_id}_meta.json"
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_csv(self, track: TrackFeatures):
        """Save as CSV file."""
        df = track.to_dataframe()
        filepath = self.base_path / f"track_{track.track_id}.csv"
        df.to_csv(filepath, index=False)
    
    def _save_json(self, track: TrackFeatures):
        """Save as JSON file."""
        filepath = self.base_path / f"track_{track.track_id}.json"
        with open(filepath, 'w') as f:
            json.dump(track.to_dict(), f, indent=2)
    
    def load_track(self, track_id: int, format: str = 'parquet') -> Optional[TrackFeatures]:
        """Load track features from storage."""
        if track_id in self.tracks_cache:
            return self.tracks_cache[track_id]
        
        if format == 'parquet':
            return self._load_parquet(track_id)
        elif format == 'csv':
            return self._load_csv(track_id)
        elif format == 'json':
            return self._load_json(track_id)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _load_parquet(self, track_id: int) -> Optional[TrackFeatures]:
        """Load from Parquet file."""
        filepath = self.base_path / f"track_{track_id}.parquet"
        meta_filepath = self.base_path / f"track_{track_id}_meta.json"
        
        if not filepath.exists():
            return None
        
        df = pd.read_parquet(filepath)
        
        # Load metadata
        metadata = {}
        if meta_filepath.exists():
            with open(meta_filepath, 'r') as f:
                metadata = json.load(f)
        
        # Reconstruct TrackFeatures
        track = self._dataframe_to_track(df, metadata)
        self.tracks_cache[track_id] = track
        return track
    
    def _load_csv(self, track_id: int) -> Optional[TrackFeatures]:
        """Load from CSV file."""
        filepath = self.base_path / f"track_{track_id}.csv"
        if not filepath.exists():
            return None
        
        df = pd.read_csv(filepath)
        track = self._dataframe_to_track(df, {})
        self.tracks_cache[track_id] = track
        return track
    
    def _load_json(self, track_id: int) -> Optional[TrackFeatures]:
        """Load from JSON file."""
        filepath = self.base_path / f"track_{track_id}.json"
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        track = TrackFeatures(**data)
        self.tracks_cache[track_id] = track
        return track
    
    def _dataframe_to_track(self, df: pd.DataFrame, metadata: Dict) -> TrackFeatures:
        """Convert DataFrame back to TrackFeatures."""
        track_id = df['track_id'].iloc[0] if 'track_id' in df.columns else metadata.get('track_id', 0)
        
        # Extract time-series data
        positions = []
        velocities = []
        
        if 'pos_x' in df.columns:
            positions = df[['pos_x', 'pos_y', 'pos_z']].values.tolist()
        
        if 'vel_x' in df.columns:
            velocities = df[['vel_x', 'vel_y', 'vel_z']].values.tolist()
        
        track = TrackFeatures(
            track_id=int(track_id),
            timestamps=df['timestamp'].tolist() if 'timestamp' in df.columns else [],
            ranges=df['range'].tolist() if 'range' in df.columns else [],
            azimuths=df['azimuth'].tolist() if 'azimuth' in df.columns else [],
            elevations=df['elevation'].tolist() if 'elevation' in df.columns else [],
            range_rates=df['range_rate'].tolist() if 'range_rate' in df.columns else [],
            positions=positions,
            velocities=velocities,
            accelerations=[],
            kalman_states=[],
            kalman_covariances=[],
            innovations=[],
            snr_values=df['snr'].tolist() if 'snr' in df.columns else [],
            rcs_values=df['rcs'].tolist() if 'rcs' in df.columns else [],
            doppler_values=df['doppler'].tolist() if 'doppler' in df.columns else [],
            pos_errors=[],
            vel_errors=[],
        )
        
        # Apply metadata if available
        for key, value in metadata.items():
            if hasattr(track, key):
                setattr(track, key, value)
        
        return track
    
    def export_all_tracks(self, output_path: str, format: str = 'csv'):
        """Export all tracks to a single file."""
        output_path = Path(output_path)
        
        all_dfs = []
        for track_id, track in self.tracks_cache.items():
            df = track.to_dataframe()
            all_dfs.append(df)
        
        if not all_dfs:
            return
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        if format == 'csv':
            combined_df.to_csv(output_path, index=False)
        elif format == 'parquet':
            combined_df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def get_all_track_ids(self) -> List[int]:
        """Get list of all stored track IDs."""
        track_ids = set()
        
        for file in self.base_path.glob("track_*.parquet"):
            track_id = int(file.stem.split('_')[1])
            track_ids.add(track_id)
        
        for file in self.base_path.glob("track_*.csv"):
            track_id = int(file.stem.split('_')[1])
            track_ids.add(track_id)
        
        return sorted(list(track_ids))
