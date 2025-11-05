"""Tests for feature store."""

import pytest
import tempfile
import shutil
from pathlib import Path

from feature_store.feature_store import FeatureStore, TrackFeatures


class TestTrackFeatures:
    """Tests for TrackFeatures."""
    
    def test_feature_creation(self):
        """Test creating track features."""
        track = TrackFeatures(
            track_id=1,
            timestamps=[0, 1, 2],
            ranges=[5000, 4950, 4900],
            azimuths=[0.5, 0.52, 0.54],
            elevations=[0.2, 0.21, 0.22],
            range_rates=[-50, -50, -50],
            positions=[[100, 200, 5000], [150, 250, 5050], [200, 300, 5100]],
            velocities=[[50, 50, 10], [50, 50, 10], [50, 50, 10]],
            accelerations=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            kalman_states=[],
            kalman_covariances=[],
            innovations=[],
            snr_values=[30, 30, 30],
            rcs_values=[10, 10, 10],
            doppler_values=[1000, 1000, 1000],
            pos_errors=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            vel_errors=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        
        assert track.track_id == 1
        assert len(track.timestamps) == 3
        assert track.flight_time > 0
    
    def test_aggregate_features(self):
        """Test aggregate feature computation."""
        track = TrackFeatures(
            track_id=1,
            timestamps=[0, 1, 2, 3],
            ranges=[5000, 4900, 4800, 4700],
            azimuths=[0.5] * 4,
            elevations=[0.2] * 4,
            range_rates=[-100] * 4,
            positions=[[0, 0, 1000], [0, 0, 2000], [0, 0, 3000], [0, 0, 4000]],
            velocities=[[100, 0, 0]] * 4,
            accelerations=[[0, 0, 0]] * 4,
            kalman_states=[],
            kalman_covariances=[],
            innovations=[],
            snr_values=[30] * 4,
            rcs_values=[10] * 4,
            doppler_values=[1000] * 4,
            pos_errors=[],
            vel_errors=[],
        )
        
        assert track.max_height == 4000
        assert track.min_height == 1000
        assert track.mean_speed == pytest.approx(100.0)
    
    def test_feature_vector(self):
        """Test feature vector generation."""
        track = TrackFeatures(
            track_id=1,
            timestamps=[0, 1],
            ranges=[5000, 4900],
            azimuths=[0.5, 0.5],
            elevations=[0.2, 0.2],
            range_rates=[-100, -100],
            positions=[[0, 0, 1000], [0, 0, 1500]],
            velocities=[[100, 0, 0], [100, 0, 0]],
            accelerations=[[0, 0, 0], [0, 0, 0]],
            kalman_states=[],
            kalman_covariances=[],
            innovations=[],
            snr_values=[30, 30],
            rcs_values=[10, 10],
            doppler_values=[1000, 1000],
            pos_errors=[],
            vel_errors=[],
        )
        
        vec = track.get_feature_vector()
        assert vec.shape == (13,)
        assert vec[0] == track.flight_time


class TestFeatureStore:
    """Tests for FeatureStore."""
    
    def test_save_and_load_parquet(self):
        """Test saving and loading with parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(tmpdir)
            
            track = TrackFeatures(
                track_id=1,
                timestamps=[0, 1, 2],
                ranges=[5000, 4900, 4800],
                azimuths=[0.5] * 3,
                elevations=[0.2] * 3,
                range_rates=[-100] * 3,
                positions=[[0, 0, 1000]] * 3,
                velocities=[[100, 0, 0]] * 3,
                accelerations=[[0, 0, 0]] * 3,
                kalman_states=[],
                kalman_covariances=[],
                innovations=[],
                snr_values=[30] * 3,
                rcs_values=[10] * 3,
                doppler_values=[1000] * 3,
                pos_errors=[],
                vel_errors=[],
            )
            
            # Save
            store.save_track(track, format='parquet')
            
            # Load
            loaded_track = store.load_track(1, format='parquet')
            
            assert loaded_track is not None
            assert loaded_track.track_id == 1
            assert len(loaded_track.timestamps) == 3
