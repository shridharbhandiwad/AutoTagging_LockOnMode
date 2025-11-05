"""
Script to generate sample datasets for testing and training.
"""
import argparse
from pathlib import Path
from simulator import TrackSimulator, SimulationConfig


def generate_training_dataset(output_dir: str, num_tracks: int = 100):
    """Generate large training dataset"""
    print(f"Generating training dataset with {num_tracks} tracks...")
    
    config = SimulationConfig(
        duration=60.0,
        dt=0.1,
        noise_level=0.1,
        dropout_rate=0.05
    )
    
    simulator = TrackSimulator(config)
    
    # Generate scenario
    scenario_df = simulator.generate_scenario(
        num_tracks=num_tracks,
        behavior_distribution={
            'linear': 0.3,
            'high_speed': 0.2,
            'maneuvering': 0.3,
            'climb': 0.2
        }
    )
    
    # Generate ground truth tags
    tags_df = simulator.generate_ground_truth_tags(scenario_df)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scenario_df.to_csv(output_path / "training_data.csv", index=False)
    tags_df.to_csv(output_path / "training_tags.csv", index=False)
    
    print(f"Saved training data to {output_path}")
    print(f"  - {len(scenario_df)} measurements")
    print(f"  - {len(tags_df)} tracks")


def generate_test_scenarios(output_dir: str):
    """Generate test scenarios with known behaviors"""
    print("Generating test scenarios...")
    
    config = SimulationConfig(duration=30.0, dt=0.1)
    simulator = TrackSimulator(config)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Scenario 1: Simple linear tracks
    print("  - Scenario 1: Linear tracks")
    scenario1 = simulator.generate_scenario(5, {'linear': 1.0})
    tags1 = simulator.generate_ground_truth_tags(scenario1)
    scenario1.to_csv(output_path / "scenario1_linear.csv", index=False)
    tags1.to_csv(output_path / "scenario1_tags.csv", index=False)
    
    # Scenario 2: High-speed tracks
    print("  - Scenario 2: High-speed tracks")
    scenario2 = simulator.generate_scenario(5, {'high_speed': 1.0})
    tags2 = simulator.generate_ground_truth_tags(scenario2)
    scenario2.to_csv(output_path / "scenario2_highspeed.csv", index=False)
    tags2.to_csv(output_path / "scenario2_tags.csv", index=False)
    
    # Scenario 3: Maneuvering tracks
    print("  - Scenario 3: Maneuvering tracks")
    scenario3 = simulator.generate_scenario(5, {'maneuvering': 1.0})
    tags3 = simulator.generate_ground_truth_tags(scenario3)
    scenario3.to_csv(output_path / "scenario3_maneuvering.csv", index=False)
    tags3.to_csv(output_path / "scenario3_tags.csv", index=False)
    
    # Scenario 4: Mixed behaviors
    print("  - Scenario 4: Mixed behaviors")
    scenario4 = simulator.generate_scenario(20, {
        'linear': 0.25,
        'high_speed': 0.25,
        'maneuvering': 0.25,
        'climb': 0.25
    })
    tags4 = simulator.generate_ground_truth_tags(scenario4)
    scenario4.to_csv(output_path / "scenario4_mixed.csv", index=False)
    tags4.to_csv(output_path / "scenario4_tags.csv", index=False)
    
    print(f"Saved test scenarios to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample datasets")
    parser.add_argument('--output-dir', default='./data/samples',
                       help='Output directory for generated data')
    parser.add_argument('--num-tracks', type=int, default=100,
                       help='Number of tracks for training dataset')
    parser.add_argument('--training', action='store_true',
                       help='Generate training dataset')
    parser.add_argument('--test', action='store_true',
                       help='Generate test scenarios')
    
    args = parser.parse_args()
    
    if args.training:
        generate_training_dataset(args.output_dir, args.num_tracks)
    
    if args.test:
        generate_test_scenarios(args.output_dir)
    
    if not args.training and not args.test:
        # Generate both by default
        generate_training_dataset(args.output_dir, args.num_tracks)
        generate_test_scenarios(args.output_dir)


if __name__ == "__main__":
    main()
