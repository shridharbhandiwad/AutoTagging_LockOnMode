#!/usr/bin/env python3
"""
Demo script showing end-to-end workflow.

This script:
1. Generates synthetic track data
2. Trains ML models
3. Runs inference
4. Shows results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.simulator import TrackSimulator, SimulationScenario
from parsers.text_parser import TextParser
from feature_store.feature_store import TrackFeatures, FeatureStore
from ml.models import RandomForestTagger, XGBoostTagger
from ml.trainer import ModelTrainer
from ml.inference import ModelInference
import numpy as np


def main():
    print("="*60)
    print("Airborne Track Behavior Tagging - Demo")
    print("="*60)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic track data...")
    scenario = SimulationScenario(
        num_tracks=20,
        duration_seconds=30.0,
        update_rate_hz=10.0,
        output_format='csv',
    )
    
    simulator = TrackSimulator(scenario)
    output_files = simulator.run_simulation('./data/demo')
    
    csv_file = next((f for f in output_files if f.endswith('.csv')), None)
    print(f"✓ Generated data: {csv_file}")
    
    # Step 2: Load and parse data
    print("\n[2/5] Loading and parsing data...")
    parser = TextParser()
    df = parser.parse_file(csv_file)
    df = TextParser.standardize_columns(df)
    
    # Convert to TrackFeatures
    tracks = []
    for track_id, track_df in df.groupby('track_id'):
        track = TrackFeatures(
            track_id=int(track_id),
            timestamps=track_df['timestamp'].tolist(),
            ranges=track_df['range'].tolist() if 'range' in track_df.columns else [],
            azimuths=track_df['azimuth'].tolist() if 'azimuth' in track_df.columns else [],
            elevations=track_df['elevation'].tolist() if 'elevation' in track_df.columns else [],
            range_rates=track_df['range_rate'].tolist() if 'range_rate' in track_df.columns else [],
            positions=track_df[['pos_x', 'pos_y', 'pos_z']].values.tolist() if 'pos_x' in track_df.columns else [],
            velocities=track_df[['vel_x', 'vel_y', 'vel_z']].values.tolist() if 'vel_x' in track_df.columns else [],
            accelerations=[],
            kalman_states=[],
            kalman_covariances=[],
            innovations=[],
            snr_values=track_df['snr'].tolist() if 'snr' in track_df.columns else [],
            rcs_values=track_df['rcs'].tolist() if 'rcs' in track_df.columns else [],
            doppler_values=track_df['doppler'].tolist() if 'doppler' in track_df.columns else [],
            pos_errors=[],
            vel_errors=[],
        )
        tracks.append(track)
    
    print(f"✓ Loaded {len(tracks)} tracks")
    
    # Step 3: Train models
    print("\n[3/5] Training ML models...")
    trainer = ModelTrainer()
    
    print("  Training Random Forest...")
    rf_model = RandomForestTagger(n_estimators=50, max_depth=8)
    rf_metrics = trainer.train_model(rf_model, tracks, test_size=0.3)
    print(f"    Overall accuracy: {rf_metrics['overall']['accuracy']:.2%}")
    
    print("  Training XGBoost...")
    xgb_model = XGBoostTagger(n_estimators=50, max_depth=5)
    xgb_metrics = trainer.train_model(xgb_model, tracks, test_size=0.3)
    print(f"    Overall accuracy: {xgb_metrics['overall']['accuracy']:.2%}")
    
    # Step 4: Run inference
    print("\n[4/5] Running inference...")
    inference = ModelInference()
    inference.add_model('RandomForest', rf_model, weight=1.0)
    inference.add_model('XGBoost', xgb_model, weight=1.5)
    
    # Apply tags to all tracks
    for track in tracks:
        inference.apply_tags_to_track(track, model_name='ensemble', threshold=0.5)
    
    print(f"✓ Tagged {len(tracks)} tracks")
    
    # Step 5: Show results
    print("\n[5/5] Results Summary:")
    print("-" * 60)
    
    for i, track in enumerate(tracks[:5]):  # Show first 5
        print(f"\nTrack {track.track_id}:")
        print(f"  Duration: {track.flight_time:.1f}s")
        print(f"  Mean Speed: {track.mean_speed:.1f} m/s")
        print(f"  Max Height: {track.max_height:.1f} m")
        print(f"  Maneuver Index: {track.maneuver_index:.3f}")
        
        if track.tags:
            print(f"  Tags:")
            for tag, confidence in sorted(track.tags.items(), 
                                         key=lambda x: x[1], reverse=True)[:3]:
                print(f"    - {tag}: {confidence:.1%}")
        else:
            print(f"  Tags: None")
    
    if len(tracks) > 5:
        print(f"\n  ... and {len(tracks) - 5} more tracks")
    
    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    
    store = FeatureStore('./data/demo/feature_store')
    for track in tracks:
        store.save_track(track, format='parquet')
    
    store.export_all_tracks('./data/demo/results.csv', format='csv')
    print("✓ Results saved to: ./data/demo/results.csv")
    
    # Model comparison
    print("\nModel Performance Comparison:")
    comparison = inference.get_model_comparison(tracks)
    for model_name, stats in comparison.items():
        print(f"  {model_name}:")
        print(f"    Avg inference time: {stats['avg_inference_time_ms']:.2f} ms")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("\nTo view in GUI:")
    print("  python -m gui.main")
    print("  Then open: ./data/demo/simulated_tracks.csv")
    print("="*60)


if __name__ == '__main__':
    main()
