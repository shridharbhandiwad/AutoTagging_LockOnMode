"""
Unit tests for ML models.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from ml.models.classical_models import ClassicalTrackTagger
from ml.feature_store import TrackFeatures, TrackTags


class TestClassicalModels:
    """Test classical ML models"""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature dataset"""
        n_samples = 100
        
        features = []
        for i in range(n_samples):
            feat = TrackFeatures(
                track_id=i,
                max_speed=np.random.uniform(100, 600),
                min_speed=np.random.uniform(50, 200),
                mean_speed=np.random.uniform(150, 400),
                std_speed=np.random.uniform(10, 50),
                max_height=np.random.uniform(3000, 10000),
                min_height=np.random.uniform(1000, 5000),
                mean_height=np.random.uniform(2000, 7000),
                max_range=np.random.uniform(8000, 20000),
                min_range=np.random.uniform(1000, 5000),
                mean_range=np.random.uniform(5000, 15000),
                maneuver_index=np.random.uniform(0, 10),
                curvature=np.random.uniform(0, 0.1),
                jerk_magnitude=np.random.uniform(0, 5),
                snr_mean=np.random.uniform(10, 30),
                snr_std=np.random.uniform(1, 5),
                rcs_mean=np.random.uniform(5, 15),
                rcs_std=np.random.uniform(0.5, 2),
                flight_time=np.random.uniform(30, 120),
                num_measurements=np.random.randint(100, 500),
                altitude_change=np.random.uniform(-1000, 1000),
                max_acceleration=np.random.uniform(0, 20),
                mean_acceleration=np.random.uniform(0, 10)
            )
            features.append(feat)
        
        return pd.DataFrame([vars(f) for f in features])
    
    @pytest.fixture
    def sample_tags(self, sample_features):
        """Create sample tags"""
        tags = []
        for _, row in sample_features.iterrows():
            tag = TrackTags(
                track_id=row['track_id'],
                high_speed=row['mean_speed'] > 400,
                low_speed=row['mean_speed'] < 200,
                high_maneuver=row['maneuver_index'] > 5,
                linear_track=row['maneuver_index'] < 2,
                climb=row['altitude_change'] > 500,
                descent=row['altitude_change'] < -500
            )
            tags.append(tag)
        
        return pd.DataFrame([vars(t) for t in tags])
    
    def test_random_forest_training(self, sample_features, sample_tags):
        """Test RandomForest model training"""
        tagger = ClassicalTrackTagger(model_name="RandomForest")
        
        metrics = tagger.train(sample_features, sample_tags, test_size=0.2)
        
        assert len(metrics) > 0
        assert 'high_speed' in metrics or 'linear_track' in metrics
        
        # Check that some models were trained
        assert len(tagger.models) > 0
    
    def test_prediction(self, sample_features, sample_tags):
        """Test model prediction"""
        tagger = ClassicalTrackTagger(model_name="RandomForest")
        tagger.train(sample_features, sample_tags, test_size=0.2)
        
        # Predict on new data
        predictions = tagger.predict(sample_features.head(10))
        
        assert len(predictions) == 10
        assert 'track_id' in predictions.columns
    
    def test_feature_importance(self, sample_features, sample_tags):
        """Test feature importance extraction"""
        tagger = ClassicalTrackTagger(model_name="RandomForest")
        tagger.train(sample_features, sample_tags, test_size=0.2)
        
        # Get feature importance
        for tag in tagger.models.keys():
            importance = tagger.get_feature_importance(tag, top_n=5)
            assert len(importance) <= 5
            assert all(isinstance(v, (int, float)) for v in importance.values())


class TestFeatureStore:
    """Test feature store"""
    
    def test_store_and_load_features(self, tmp_path):
        """Test storing and loading features"""
        from ml.feature_store import FeatureStore
        
        store = FeatureStore(str(tmp_path))
        
        # Create sample features
        features = [
            TrackFeatures(
                track_id=1,
                max_speed=500, min_speed=100, mean_speed=300, std_speed=50,
                max_height=5000, min_height=3000, mean_height=4000,
                max_range=10000, min_range=5000, mean_range=7500,
                maneuver_index=2.5, curvature=0.05, jerk_magnitude=1.0,
                snr_mean=20, snr_std=2, rcs_mean=10, rcs_std=1,
                flight_time=60, num_measurements=600
            )
        ]
        
        # Store features
        store.store_features(features)
        
        # Load features
        loaded = store.load_features()
        
        assert len(loaded) == 1
        assert loaded['track_id'].iloc[0] == 1
        assert loaded['mean_speed'].iloc[0] == 300
    
    def test_csv_export(self, tmp_path):
        """Test CSV export"""
        from ml.feature_store import FeatureStore, TrackFeatures
        
        store = FeatureStore(str(tmp_path / "store"))
        
        features = [
            TrackFeatures(
                track_id=i,
                max_speed=500, min_speed=100, mean_speed=300, std_speed=50,
                max_height=5000, min_height=3000, mean_height=4000,
                max_range=10000, min_range=5000, mean_range=7500,
                maneuver_index=2.5, curvature=0.05, jerk_magnitude=1.0,
                snr_mean=20, snr_std=2, rcs_mean=10, rcs_std=1,
                flight_time=60, num_measurements=600
            )
            for i in range(5)
        ]
        
        store.store_features(features)
        
        # Export
        export_dir = tmp_path / "export"
        exported = store.export_to_csv(str(export_dir))
        
        assert 'features' in exported
        assert Path(exported['features']).exists()
