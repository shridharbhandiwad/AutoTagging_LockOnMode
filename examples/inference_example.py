"""
Example script for running inference on track data.
"""
import pandas as pd
from ml import InferencePipeline
from ml.inference.explainability import ExplainabilityAnalyzer
from parsers import FileRouter


def main():
    print("=== Track Behavior Tagger - Inference Example ===\n")
    
    # Step 1: Load models
    print("Step 1: Loading trained models...")
    pipeline = InferencePipeline()
    
    try:
        pipeline.load_models('./models')
        print(f"  Loaded {len(pipeline.models)} models: {', '.join(pipeline.models.keys())}")
    except FileNotFoundError:
        print("  Error: Models not found. Please run train_example.py first.")
        return
    
    # Step 2: Load or generate test data
    print("\nStep 2: Loading test data...")
    
    # Try to load from samples
    try:
        scenario_df = pd.read_csv('./data/samples/scenario1_linear.csv')
        print(f"  Loaded {len(scenario_df)} measurements")
    except FileNotFoundError:
        print("  Warning: Sample data not found. Generating test data...")
        from simulator import TrackSimulator
        simulator = TrackSimulator()
        scenario_df = simulator.generate_scenario(num_tracks=5, behavior_distribution={'linear': 1.0})
        print(f"  Generated {len(scenario_df)} measurements")
    
    # Step 3: Extract features
    print("\nStep 3: Extracting features...")
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
    print(f"  Extracted {len(features_df)} track features")
    
    # Step 4: Run inference
    print("\nStep 4: Running inference...")
    
    # Predict with each model
    for model_name in pipeline.models.keys():
        print(f"\n  Using {model_name}:")
        predictions = pipeline.predict(features_df, model_name=model_name)
        
        # Show results
        for _, row in predictions.head(3).iterrows():
            print(f"    Track {int(row['track_id'])}:")
            
            # Show only positive tags
            tags = []
            for tag in ['high_speed', 'low_speed', 'high_maneuver', 'linear_track', 'climb', 'descent']:
                if row.get(tag, False):
                    conf = row.get(f'{tag}_conf', 0.0)
                    tags.append(f"{tag} ({conf:.2f})")
            
            if tags:
                print(f"      Tags: {', '.join(tags)}")
            else:
                print(f"      Tags: (none)")
    
    # Step 5: Ensemble prediction
    if len(pipeline.models) > 1:
        print("\n  Using Ensemble:")
        ensemble_predictions = pipeline.predict(features_df, ensemble=True)
        
        for _, row in ensemble_predictions.head(3).iterrows():
            print(f"    Track {int(row['track_id'])}:")
            
            tags = []
            for tag in ['high_speed', 'low_speed', 'high_maneuver', 'linear_track', 'climb', 'descent']:
                if row.get(tag, False):
                    conf = row.get(f'{tag}_conf', 0.0)
                    tags.append(f"{tag} ({conf:.2f})")
            
            if tags:
                print(f"      Tags: {', '.join(tags)}")
            else:
                print(f"      Tags: (none)")
    
    # Step 6: Explain predictions (if RandomForest available)
    if 'RandomForest' in pipeline.models:
        print("\nStep 5: Explaining predictions...")
        from ml.models.classical_models import ClassicalTrackTagger
        
        model = pipeline.models['RandomForest']
        if isinstance(model, ClassicalTrackTagger):
            # Show feature importance
            print("\n  Feature Importance for 'high_speed' tag:")
            importance = model.get_feature_importance('high_speed', top_n=5)
            
            for feature, value in importance.items():
                print(f"    {feature}: {value:.4f}")
    
    print("\n=== Inference complete! ===")


if __name__ == "__main__":
    main()
