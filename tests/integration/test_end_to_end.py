"""
Integration tests for end-to-end workflows.
"""
import pytest
import tempfile
from pathlib import Path
import pandas as pd

from simulator import TrackSimulator, SimulationConfig
from parsers import FileRouter
from ml import FeatureStore, ClassicalTrackTagger, TrainingPipeline, InferencePipeline


class TestEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def sample_scenario(self, tmp_path):
        """Generate sample scenario"""
        config = SimulationConfig(duration=30.0, dt=0.5)
        simulator = TrackSimulator(config)
        
        scenario_df = simulator.generate_scenario(num_tracks=10)
        tags_df = simulator.generate_ground_truth_tags(scenario_df)
        
        # Save to files
        scenario_file = tmp_path / "scenario.csv"
        tags_file = tmp_path / "tags.csv"
        
        scenario_df.to_csv(scenario_file, index=False)
        tags_df.to_csv(tags_file, index=False)
        
        return scenario_file, tags_file, scenario_df, tags_df
    
    def test_simulation_to_parsing(self, sample_scenario):
        """Test: Generate data → Parse file"""
        scenario_file, _, _, _ = sample_scenario
        
        # Parse file
        parsed_data = FileRouter.parse_file(str(scenario_file))
        
        assert not parsed_data.empty
        assert 'track_id' in parsed_data.columns
        assert 'timestamp' in parsed_data.columns
    
    def test_simulation_to_feature_extraction(self, sample_scenario, tmp_path):
        """Test: Generate data → Extract features → Store"""
        scenario_file, _, scenario_df, _ = sample_scenario
        
        # Initialize feature store
        store = FeatureStore(str(tmp_path / "feature_store"))
        
        # Extract features (simplified)
        from ml.feature_store import TrackFeatures
        
        features = []
        for track_id in scenario_df['track_id'].unique():
            track_data = scenario_df[scenario_df['track_id'] == track_id]
            
            speeds = (track_data['vel_x']**2 + track_data['vel_y']**2 + 
                     track_data['vel_z']**2)**0.5
            
            feat = TrackFeatures(
                track_id=int(track_id),
                max_speed=float(speeds.max()),
                min_speed=float(speeds.min()),
                mean_speed=float(speeds.mean()),
                std_speed=float(speeds.std()),
                max_height=float(track_data['pos_z'].max()),
                min_height=float(track_data['pos_z'].min()),
                mean_height=float(track_data['pos_z'].mean()),
                max_range=0.0, min_range=0.0, mean_range=0.0,
                maneuver_index=0.0, curvature=0.0, jerk_magnitude=0.0,
                snr_mean=float(track_data['snr'].mean()),
                snr_std=float(track_data['snr'].std()),
                rcs_mean=float(track_data['rcs'].mean()),
                rcs_std=float(track_data['rcs'].std()),
                flight_time=float(track_data['timestamp'].max() - track_data['timestamp'].min()),
                num_measurements=len(track_data)
            )
            features.append(feat)
        
        # Store features
        store.store_features(features)
        
        # Load and verify
        loaded_features = store.load_features()
        assert len(loaded_features) == len(features)
    
    def test_full_training_pipeline(self, sample_scenario, tmp_path):
        """Test: Generate data → Train models → Save"""
        _, tags_file, scenario_df, tags_df = sample_scenario
        
        # Extract features
        from ml.feature_store import TrackFeatures
        
        features = []
        for track_id in scenario_df['track_id'].unique():
            track_data = scenario_df[scenario_df['track_id'] == track_id]
            
            speeds = (track_data['vel_x']**2 + track_data['vel_y']**2 + 
                     track_data['vel_z']**2)**0.5
            
            accels = (track_data['acc_x']**2 + track_data['acc_y']**2 + 
                     track_data['acc_z']**2)**0.5
            
            feat = TrackFeatures(
                track_id=int(track_id),
                max_speed=float(speeds.max()),
                min_speed=float(speeds.min()),
                mean_speed=float(speeds.mean()),
                std_speed=float(speeds.std()),
                max_height=float(track_data['pos_z'].max()),
                min_height=float(track_data['pos_z'].min()),
                mean_height=float(track_data['pos_z'].mean()),
                max_range=0.0, min_range=0.0, mean_range=0.0,
                maneuver_index=float(accels.mean()),
                curvature=0.0,
                jerk_magnitude=0.0,
                snr_mean=float(track_data['snr'].mean()),
                snr_std=float(track_data['snr'].std()),
                rcs_mean=float(track_data['rcs'].mean()),
                rcs_std=float(track_data['rcs'].std()),
                flight_time=float(track_data['timestamp'].max() - track_data['timestamp'].min()),
                num_measurements=len(track_data),
                altitude_change=float(track_data['pos_z'].iloc[-1] - track_data['pos_z'].iloc[0]),
                max_acceleration=float(accels.max()),
                mean_acceleration=float(accels.mean())
            )
            features.append(feat)
        
        features_df = pd.DataFrame([vars(f) for f in features])
        
        # Initialize feature store and training pipeline
        store = FeatureStore(str(tmp_path / "feature_store"))
        pipeline = TrainingPipeline(store)
        
        # Train models (just RandomForest for speed)
        results = pipeline.train_classical_models(
            features_df, tags_df,
            model_types=['RandomForest'],
            test_size=0.3,
            cv_folds=2
        )
        
        assert 'RandomForest' in results
        
        # Save models
        models_dir = tmp_path / "models"
        pipeline.save_models(str(models_dir))
        
        assert (models_dir / "training_metrics.json").exists()
    
    def test_inference_pipeline(self, sample_scenario, tmp_path):
        """Test: Train → Save → Load → Predict"""
        _, tags_file, scenario_df, tags_df = sample_scenario
        
        # Extract features (same as above)
        from ml.feature_store import TrackFeatures
        
        features = []
        for track_id in scenario_df['track_id'].unique():
            track_data = scenario_df[scenario_df['track_id'] == track_id]
            
            speeds = (track_data['vel_x']**2 + track_data['vel_y']**2 + 
                     track_data['vel_z']**2)**0.5
            accels = (track_data['acc_x']**2 + track_data['acc_y']**2 + 
                     track_data['acc_z']**2)**0.5
            
            feat = TrackFeatures(
                track_id=int(track_id),
                max_speed=float(speeds.max()),
                min_speed=float(speeds.min()),
                mean_speed=float(speeds.mean()),
                std_speed=float(speeds.std()),
                max_height=float(track_data['pos_z'].max()),
                min_height=float(track_data['pos_z'].min()),
                mean_height=float(track_data['pos_z'].mean()),
                max_range=0.0, min_range=0.0, mean_range=0.0,
                maneuver_index=float(accels.mean()),
                curvature=0.0, jerk_magnitude=0.0,
                snr_mean=float(track_data['snr'].mean()),
                snr_std=float(track_data['snr'].std()),
                rcs_mean=float(track_data['rcs'].mean()),
                rcs_std=float(track_data['rcs'].std()),
                flight_time=float(track_data['timestamp'].max() - track_data['timestamp'].min()),
                num_measurements=len(track_data),
                altitude_change=float(track_data['pos_z'].iloc[-1] - track_data['pos_z'].iloc[0]),
                max_acceleration=float(accels.max()),
                mean_acceleration=float(accels.mean())
            )
            features.append(feat)
        
        features_df = pd.DataFrame([vars(f) for f in features])
        
        # Train and save
        store = FeatureStore(str(tmp_path / "feature_store"))
        train_pipeline = TrainingPipeline(store)
        
        train_pipeline.train_classical_models(
            features_df, tags_df,
            model_types=['RandomForest'],
            test_size=0.3,
            cv_folds=1
        )
        
        models_dir = tmp_path / "models"
        train_pipeline.save_models(str(models_dir))
        
        # Load and infer
        infer_pipeline = InferencePipeline()
        infer_pipeline.load_models(str(models_dir))
        
        # Predict on same data
        predictions = infer_pipeline.predict(features_df.head(5))
        
        assert not predictions.empty
        assert 'track_id' in predictions.columns
