#!/usr/bin/env python3
"""Script to train ML models on track data."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.simulator import TrackSimulator, SimulationScenario
from feature_store.feature_store import FeatureStore, TrackFeatures
from ml.models import RandomForestTagger, XGBoostTagger, LSTMTagger
from ml.trainer import ModelTrainer
from ml.explainability import ExplainabilityAnalyzer
import numpy as np


def generate_training_data(num_tracks: int = 100, output_dir: str = './data/training'):
    """Generate synthetic training data."""
    print(f"Generating {num_tracks} tracks for training...")
    
    scenario = SimulationScenario(
        num_tracks=num_tracks,
        duration_seconds=30.0,
        update_rate_hz=10.0,
        output_format='csv',
    )
    
    simulator = TrackSimulator(scenario)
    output_files = simulator.run_simulation(output_dir)
    
    print(f"Generated files: {output_files}")
    return output_files


def load_tracks_from_csv(csv_file: str) -> list:
    """Load tracks from CSV file."""
    import pandas as pd
    from parsers.text_parser import TextParser
    
    print(f"Loading tracks from {csv_file}...")
    
    parser = TextParser()
    df = parser.parse_file(csv_file)
    df = TextParser.standardize_columns(df)
    
    # Group by track_id
    tracks = []
    for track_id, track_df in df.groupby('track_id'):
        # Convert to TrackFeatures
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
    
    print(f"Loaded {len(tracks)} tracks")
    return tracks


def train_random_forest(tracks: list, output_dir: str):
    """Train Random Forest model."""
    print("\n" + "="*50)
    print("Training Random Forest Model")
    print("="*50)
    
    trainer = ModelTrainer()
    model = RandomForestTagger(n_estimators=100, max_depth=10)
    
    metrics = trainer.train_model(model, tracks, test_size=0.2)
    
    print("\nMetrics:")
    for tag, tag_metrics in metrics.items():
        if tag != 'overall':
            print(f"  {tag}:")
            for metric_name, value in tag_metrics.items():
                print(f"    {metric_name}: {value:.4f}")
    
    # Save model
    model_path = Path(output_dir) / 'random_forest'
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    import json
    with open(model_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate feature importance plot
    X, _ = trainer.prepare_data(tracks)
    analyzer = ExplainabilityAnalyzer(model, trainer.feature_names)
    
    for tag in ['high_speed', 'high_maneuver']:
        try:
            plot_path = model_path / f'importance_{tag}.png'
            analyzer.plot_feature_importance(tag, str(plot_path))
            print(f"Feature importance plot saved: {plot_path}")
        except Exception as e:
            print(f"Warning: Could not generate plot for {tag}: {e}")
    
    return model


def train_xgboost(tracks: list, output_dir: str):
    """Train XGBoost model."""
    print("\n" + "="*50)
    print("Training XGBoost Model")
    print("="*50)
    
    trainer = ModelTrainer()
    model = XGBoostTagger(n_estimators=100, max_depth=6, learning_rate=0.1)
    
    metrics = trainer.train_model(model, tracks, test_size=0.2)
    
    print("\nMetrics:")
    for tag, tag_metrics in metrics.items():
        if tag != 'overall':
            print(f"  {tag}:")
            for metric_name, value in tag_metrics.items():
                print(f"    {metric_name}: {value:.4f}")
    
    # Save model
    model_path = Path(output_dir) / 'xgboost'
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    import json
    with open(model_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return model


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train ML models for track behavior tagging')
    parser.add_argument('--data', type=str, help='Path to training data CSV (if not provided, generates synthetic data)')
    parser.add_argument('--num-tracks', type=int, default=100, help='Number of tracks to generate if --data not provided')
    parser.add_argument('--output-dir', type=str, default='./models/saved', help='Output directory for trained models')
    parser.add_argument('--models', nargs='+', default=['rf', 'xgb'], 
                       choices=['rf', 'xgb', 'lstm'], help='Models to train')
    
    args = parser.parse_args()
    
    # Get training data
    if args.data:
        tracks = load_tracks_from_csv(args.data)
    else:
        print("No data file provided, generating synthetic data...")
        output_files = generate_training_data(args.num_tracks)
        
        # Find CSV file
        csv_file = next((f for f in output_files if f.endswith('.csv')), None)
        if not csv_file:
            print("Error: No CSV file generated")
            return
        
        tracks = load_tracks_from_csv(csv_file)
    
    if len(tracks) == 0:
        print("Error: No tracks to train on")
        return
    
    # Train models
    if 'rf' in args.models:
        train_random_forest(tracks, args.output_dir)
    
    if 'xgb' in args.models:
        train_xgboost(tracks, args.output_dir)
    
    if 'lstm' in args.models:
        print("\nNote: LSTM training requires sequence data and is more complex.")
        print("Skipping LSTM for now. Implement sequence preparation for LSTM training.")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == '__main__':
    main()
