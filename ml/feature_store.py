"""
Feature store for persisting track measurements, processed data, and derived features.
Supports CSV and Parquet formats with efficient querying.
"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class TrackMeasurement:
    """Raw measurement record"""
    track_id: int
    timestamp: float
    range: float
    azimuth: float
    elevation: float
    range_rate: float
    snr: float
    rcs: float
    doppler: float
    

@dataclass
class ProcessedTrackState:
    """Processed track state from Kalman filter"""
    track_id: int
    timestamp: float
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    pos_error_x: float = 0.0
    pos_error_y: float = 0.0
    pos_error_z: float = 0.0
    vel_error_x: float = 0.0
    vel_error_y: float = 0.0
    vel_error_z: float = 0.0
    residual_x: float = 0.0
    residual_y: float = 0.0
    residual_z: float = 0.0
    innovation_magnitude: float = 0.0


@dataclass
class TrackFeatures:
    """Derived features for ML"""
    track_id: int
    
    # Speed statistics
    max_speed: float
    min_speed: float
    mean_speed: float
    std_speed: float
    
    # Height statistics
    max_height: float
    min_height: float
    mean_height: float
    
    # Range statistics
    max_range: float
    min_range: float
    mean_range: float
    
    # Maneuver indicators
    maneuver_index: float
    curvature: float
    jerk_magnitude: float
    
    # Signal quality
    snr_mean: float
    snr_std: float
    rcs_mean: float
    rcs_std: float
    
    # Temporal
    flight_time: float
    num_measurements: int
    
    # Additional derived features
    altitude_change: float = 0.0
    max_acceleration: float = 0.0
    mean_acceleration: float = 0.0
    

@dataclass
class TrackTags:
    """Behavior tags assigned by ML models"""
    track_id: int
    
    # Binary tags
    high_speed: bool = False
    low_speed: bool = False
    high_maneuver: bool = False
    linear_track: bool = False
    climb: bool = False
    descent: bool = False
    hover_like: bool = False
    two_jet: bool = False
    multiengine: bool = False
    
    # Confidence scores (0-1)
    high_speed_conf: float = 0.0
    low_speed_conf: float = 0.0
    high_maneuver_conf: float = 0.0
    linear_track_conf: float = 0.0
    climb_conf: float = 0.0
    descent_conf: float = 0.0
    
    # Model metadata
    model_name: str = ""
    model_version: str = "1.0"
    inference_time_ms: float = 0.0


class FeatureStore:
    """Store and manage track data, features, and tags"""
    
    def __init__(self, base_path: str = "./data/feature_store"):
        """
        Initialize feature store.
        
        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Separate paths for different data types
        self.measurements_path = self.base_path / "measurements"
        self.processed_path = self.base_path / "processed"
        self.features_path = self.base_path / "features"
        self.tags_path = self.base_path / "tags"
        
        for path in [self.measurements_path, self.processed_path, 
                     self.features_path, self.tags_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def store_measurements(self, measurements: List[TrackMeasurement], 
                          format: str = "parquet") -> str:
        """
        Store raw measurements.
        
        Args:
            measurements: List of measurement records
            format: Storage format (csv/parquet)
            
        Returns:
            Path to stored file
        """
        df = pd.DataFrame([asdict(m) for m in measurements])
        
        # Group by track_id for efficient storage
        track_ids = df['track_id'].unique()
        
        for track_id in track_ids:
            track_df = df[df['track_id'] == track_id]
            filepath = self.measurements_path / f"track_{track_id}_measurements.{format}"
            
            if format == "parquet":
                track_df.to_parquet(filepath, index=False, engine='pyarrow')
            else:
                track_df.to_csv(filepath, index=False)
        
        return str(self.measurements_path)
    
    def store_processed_states(self, states: List[ProcessedTrackState],
                               format: str = "parquet") -> str:
        """Store processed track states"""
        df = pd.DataFrame([asdict(s) for s in states])
        
        track_ids = df['track_id'].unique()
        
        for track_id in track_ids:
            track_df = df[df['track_id'] == track_id]
            filepath = self.processed_path / f"track_{track_id}_processed.{format}"
            
            if format == "parquet":
                track_df.to_parquet(filepath, index=False, engine='pyarrow')
            else:
                track_df.to_csv(filepath, index=False)
        
        return str(self.processed_path)
    
    def store_features(self, features: List[TrackFeatures],
                      format: str = "parquet") -> str:
        """Store extracted features"""
        df = pd.DataFrame([asdict(f) for f in features])
        
        filepath = self.features_path / f"all_features.{format}"
        
        if format == "parquet":
            df.to_parquet(filepath, index=False, engine='pyarrow')
        else:
            df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def store_tags(self, tags: List[TrackTags],
                  format: str = "parquet") -> str:
        """Store ML-generated tags"""
        df = pd.DataFrame([asdict(t) for t in tags])
        
        filepath = self.tags_path / f"tags_{tags[0].model_name if tags else 'unknown'}.{format}"
        
        if format == "parquet":
            df.to_parquet(filepath, index=False, engine='pyarrow')
        else:
            df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def load_measurements(self, track_id: Optional[int] = None) -> pd.DataFrame:
        """Load measurements for specific track or all tracks"""
        if track_id is not None:
            # Load specific track
            for ext in ['parquet', 'csv']:
                filepath = self.measurements_path / f"track_{track_id}_measurements.{ext}"
                if filepath.exists():
                    if ext == 'parquet':
                        return pd.read_parquet(filepath)
                    else:
                        return pd.read_csv(filepath)
            return pd.DataFrame()
        else:
            # Load all tracks
            dfs = []
            for filepath in self.measurements_path.glob("track_*_measurements.*"):
                if filepath.suffix == '.parquet':
                    dfs.append(pd.read_parquet(filepath))
                elif filepath.suffix == '.csv':
                    dfs.append(pd.read_csv(filepath))
            
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def load_features(self) -> pd.DataFrame:
        """Load all extracted features"""
        for ext in ['parquet', 'csv']:
            filepath = self.features_path / f"all_features.{ext}"
            if filepath.exists():
                if ext == 'parquet':
                    return pd.read_parquet(filepath)
                else:
                    return pd.read_csv(filepath)
        
        return pd.DataFrame()
    
    def load_tags(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Load tags from specific model or all models"""
        if model_name:
            for ext in ['parquet', 'csv']:
                filepath = self.tags_path / f"tags_{model_name}.{ext}"
                if filepath.exists():
                    if ext == 'parquet':
                        return pd.read_parquet(filepath)
                    else:
                        return pd.read_csv(filepath)
            return pd.DataFrame()
        else:
            # Load all tags
            dfs = []
            for filepath in self.tags_path.glob("tags_*.*"):
                if filepath.suffix == '.parquet':
                    dfs.append(pd.read_parquet(filepath))
                elif filepath.suffix == '.csv':
                    dfs.append(pd.read_csv(filepath))
            
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def export_to_csv(self, output_dir: str) -> Dict[str, str]:
        """
        Export all data to CSV format.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary of exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        
        # Export measurements
        measurements_df = self.load_measurements()
        if not measurements_df.empty:
            meas_file = output_path / "measurements.csv"
            measurements_df.to_csv(meas_file, index=False)
            exported['measurements'] = str(meas_file)
        
        # Export features
        features_df = self.load_features()
        if not features_df.empty:
            feat_file = output_path / "features.csv"
            features_df.to_csv(feat_file, index=False)
            exported['features'] = str(feat_file)
        
        # Export tags
        tags_df = self.load_tags()
        if not tags_df.empty:
            tags_file = output_path / "tags.csv"
            tags_df.to_csv(tags_file, index=False)
            exported['tags'] = str(tags_file)
        
        return exported
    
    def get_track_summary(self) -> pd.DataFrame:
        """Get summary statistics for all tracks"""
        features_df = self.load_features()
        tags_df = self.load_tags()
        
        if features_df.empty:
            return pd.DataFrame()
        
        summary = features_df.copy()
        
        if not tags_df.empty:
            # Merge with tags
            tags_latest = tags_df.sort_values('inference_time_ms').groupby('track_id').last().reset_index()
            summary = summary.merge(tags_latest, on='track_id', how='left')
        
        return summary
