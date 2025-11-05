"""Tests for ML models."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from ml.models import RandomForestTagger, XGBoostTagger
from ml.trainer import ModelTrainer
from feature_store.feature_store import TrackFeatures


class TestModels:
    """Tests for ML models."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 13).astype(np.float32)
        y = {
            'high_speed': np.random.randint(0, 2, 100),
            'low_speed': np.random.randint(0, 2, 100),
            'high_maneuver': np.random.randint(0, 2, 100),
        }
        return X, y
    
    def test_random_forest_training(self, sample_data):
        """Test Random Forest training."""
        X, y = sample_data
        
        model = RandomForestTagger(n_estimators=10, max_depth=5)
        model.fit(X, y)
        
        assert model.is_fitted
        assert len(model.models) == 3
    
    def test_random_forest_prediction(self, sample_data):
        """Test Random Forest prediction."""
        X, y = sample_data
        
        model = RandomForestTagger(n_estimators=10, max_depth=5)
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        
        assert 'high_speed' in predictions
        assert len(predictions['high_speed']) == 10
        assert all(0 <= p <= 1 for p in predictions['high_speed'])
    
    def test_xgboost_training(self, sample_data):
        """Test XGBoost training."""
        X, y = sample_data
        
        model = XGBoostTagger(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        assert model.is_fitted
        assert len(model.models) == 3
    
    def test_model_save_load(self, sample_data):
        """Test model save and load."""
        X, y = sample_data
        
        model = RandomForestTagger(n_estimators=10, max_depth=5)
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save(tmpdir)
            
            # Load
            loaded_model = RandomForestTagger()
            loaded_model.load(tmpdir)
            
            assert loaded_model.is_fitted
            assert len(loaded_model.models) == 3
            
            # Compare predictions
            pred_original = model.predict(X[:5])
            pred_loaded = loaded_model.predict(X[:5])
            
            for tag in pred_original.keys():
                np.testing.assert_allclose(
                    pred_original[tag],
                    pred_loaded[tag],
                    rtol=1e-5
                )


class TestModelTrainer:
    """Tests for ModelTrainer."""
    
    @pytest.fixture
    def sample_tracks(self):
        """Generate sample tracks."""
        tracks = []
        for i in range(20):
            track = TrackFeatures(
                track_id=i,
                timestamps=list(range(10)),
                ranges=[5000 - j * 10 for j in range(10)],
                azimuths=[0.5] * 10,
                elevations=[0.2] * 10,
                range_rates=[-50] * 10,
                positions=[[0, 0, 1000 + j * 100] for j in range(10)],
                velocities=[[100 + np.random.randn() * 10, 0, 0] for j in range(10)],
                accelerations=[[0, 0, 0]] * 10,
                kalman_states=[],
                kalman_covariances=[],
                innovations=[],
                snr_values=[30] * 10,
                rcs_values=[10 + np.random.randn() for _ in range(10)],
                doppler_values=[1000] * 10,
                pos_errors=[],
                vel_errors=[],
            )
            tracks.append(track)
        return tracks
    
    def test_prepare_data(self, sample_tracks):
        """Test data preparation."""
        trainer = ModelTrainer()
        X, y = trainer.prepare_data(sample_tracks)
        
        assert X.shape[0] == 20
        assert X.shape[1] == 13
        assert len(y) > 0
    
    def test_train_model(self, sample_tracks):
        """Test model training."""
        trainer = ModelTrainer()
        model = RandomForestTagger(n_estimators=10, max_depth=5)
        
        metrics = trainer.train_model(model, sample_tracks, test_size=0.3)
        
        assert 'overall' in metrics
        assert 'accuracy' in metrics['overall']
