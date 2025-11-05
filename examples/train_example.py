"""
Example script for training track behavior tagging models.
"""
import pandas as pd
from pathlib import Path
from ml import TrainingPipeline, FeatureStore, InferencePipeline
from simulator import TrackSimulator, SimulationConfig


def main():
    print("=== Track Behavior Tagger - Training Example ===\n")
    
    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic training data...")
    config = SimulationConfig(duration=60.0, dt=0.1)
    simulator = TrackSimulator(config)
    
    scenario_df = simulator.generate_scenario(
        num_tracks=100,
        behavior_distribution={
            'linear': 0.3,
            'high_speed': 0.2,
            'maneuvering': 0.3,
            'climb': 0.2
        }
    )
    
    tags_df = simulator.generate_ground_truth_tags(scenario_df)
    print(f"  Generated {len(scenario_df)} measurements for {len(tags_df)} tracks")
    
    # Step 2: Extract features
    print("\nStep 2: Extracting features...")
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
    print(f"  Extracted features for {len(features_df)} tracks")
    
    # Step 3: Initialize feature store
    print("\nStep 3: Storing features...")
    store = FeatureStore('./data/feature_store')
    store.store_features(features)
    print("  Features stored successfully")
    
    # Step 4: Train models
    print("\nStep 4: Training models...")
    pipeline = TrainingPipeline(store)
    
    print("  Training RandomForest...")
    results_rf = pipeline.train_classical_models(
        features_df, tags_df,
        model_types=['RandomForest'],
        test_size=0.2,
        cv_folds=3
    )
    
    print("\n  Training XGBoost...")
    results_xgb = pipeline.train_classical_models(
        features_df, tags_df,
        model_types=['XGBoost'],
        test_size=0.2,
        cv_folds=3
    )
    
    # Step 5: Compare models
    print("\nStep 5: Comparing models...")
    comparison = pipeline.compare_models()
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    
    # Step 6: Save models
    print("\nStep 6: Saving models...")
    models_dir = './models'
    Path(models_dir).mkdir(exist_ok=True)
    pipeline.save_models(models_dir)
    print(f"  Models saved to {models_dir}/")
    
    # Step 7: Test inference
    print("\nStep 7: Testing inference...")
    inference_pipeline = InferencePipeline()
    inference_pipeline.load_models(models_dir)
    
    # Predict on test data
    test_features = features_df.head(10)
    predictions = inference_pipeline.predict(test_features, model_name='RandomForest')
    
    print("\nSample Predictions:")
    for _, row in predictions.head(5).iterrows():
        print(f"  Track {int(row['track_id'])}:")
        print(f"    high_speed: {row['high_speed']} (conf: {row['high_speed_conf']:.2f})")
        print(f"    high_maneuver: {row['high_maneuver']} (conf: {row['high_maneuver_conf']:.2f})")
        print(f"    linear_track: {row['linear_track']} (conf: {row['linear_track_conf']:.2f})")
    
    print("\n=== Training complete! ===")
    print(f"\nModels saved to: {models_dir}/")
    print("To use the GUI: python gui/app.py")


if __name__ == "__main__":
    main()
