"""Background processing thread for file loading."""

from PySide6.QtCore import QThread, Signal
import traceback

from parsers.file_detector import FileDetector
from parsers.binary_parser import TrackRecordParser
from parsers.text_parser import TextParser
from feature_store.feature_store import FeatureStore, TrackFeatures
import numpy as np


class ProcessingThread(QThread):
    """Thread for processing files in background."""
    
    progress = Signal(str)
    finished_signal = Signal(dict, object)  # tracks, feature_store
    error = Signal(str)
    
    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
    
    def run(self):
        """Process file."""
        try:
            self.progress.emit("Detecting file type...")
            
            # Detect file type
            detector = FileDetector()
            file_type, _ = detector.detect_file_type(self.filepath)
            
            self.progress.emit(f"Parsing {file_type} file...")
            
            # Parse file
            if file_type == 'binary':
                parser = TrackRecordParser()
                df = parser.parse_to_dataframe(self.filepath)
            else:
                parser = TextParser()
                df = parser.parse_file(self.filepath)
                df = TextParser.standardize_columns(df)
            
            self.progress.emit("Extracting tracks...")
            
            # Group by track_id
            if 'track_id' in df.columns:
                track_groups = df.groupby('track_id')
            else:
                # Single track
                track_groups = [(0, df)]
            
            # Convert to TrackFeatures
            tracks = {}
            feature_store = FeatureStore()
            
            for track_id, track_df in track_groups:
                track_id = int(track_id)
                
                self.progress.emit(f"Processing track {track_id}...")
                
                track_features = self._dataframe_to_track_features(track_id, track_df)
                
                tracks[track_id] = track_features
                feature_store.save_track(track_features, format='parquet')
            
            self.progress.emit("Complete")
            self.finished_signal.emit(tracks, feature_store)
            
        except Exception as e:
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)
    
    def _dataframe_to_track_features(self, track_id: int, df) -> TrackFeatures:
        """Convert DataFrame to TrackFeatures."""
        # Extract fields
        timestamps = df['timestamp'].tolist() if 'timestamp' in df.columns else list(range(len(df)))
        
        ranges = df['range'].tolist() if 'range' in df.columns else []
        azimuths = df['azimuth'].tolist() if 'azimuth' in df.columns else []
        elevations = df['elevation'].tolist() if 'elevation' in df.columns else []
        range_rates = df['range_rate'].tolist() if 'range_rate' in df.columns else []
        
        # Positions
        positions = []
        if 'pos_x' in df.columns:
            positions = df[['pos_x', 'pos_y', 'pos_z']].fillna(0).values.tolist()
        elif 'position_0' in df.columns:
            positions = df[['position_0', 'position_1', 'position_2']].fillna(0).values.tolist()
        
        # Velocities
        velocities = []
        if 'vel_x' in df.columns:
            velocities = df[['vel_x', 'vel_y', 'vel_z']].fillna(0).values.tolist()
        elif 'velocity_0' in df.columns:
            velocities = df[['velocity_0', 'velocity_1', 'velocity_2']].fillna(0).values.tolist()
        
        # Accelerations
        accelerations = []
        if 'acceleration_0' in df.columns:
            accelerations = df[['acceleration_0', 'acceleration_1', 'acceleration_2']].fillna(0).values.tolist()
        
        # Signal data
        snr_values = df['snr'].tolist() if 'snr' in df.columns else []
        rcs_values = df['rcs'].tolist() if 'rcs' in df.columns else []
        doppler_values = df['doppler'].tolist() if 'doppler' in df.columns else []
        
        # Errors
        pos_errors = []
        vel_errors = []
        if 'pos_error_0' in df.columns:
            pos_errors = df[['pos_error_0', 'pos_error_1', 'pos_error_2']].fillna(0).values.tolist()
        if 'vel_error_0' in df.columns:
            vel_errors = df[['vel_error_0', 'vel_error_1', 'vel_error_2']].fillna(0).values.tolist()
        
        # Create TrackFeatures
        track_features = TrackFeatures(
            track_id=track_id,
            timestamps=timestamps,
            ranges=ranges,
            azimuths=azimuths,
            elevations=elevations,
            range_rates=range_rates,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            kalman_states=[],
            kalman_covariances=[],
            innovations=[],
            snr_values=snr_values,
            rcs_values=rcs_values,
            doppler_values=doppler_values,
            pos_errors=pos_errors,
            vel_errors=vel_errors,
        )
        
        return track_features
