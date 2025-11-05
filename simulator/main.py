"""Simulator command-line interface."""

import argparse
from .simulator import TrackSimulator, SimulationScenario


def main():
    """Run simulator from command line."""
    parser = argparse.ArgumentParser(description='Track Data Simulator')
    parser.add_argument('--output-dir', type=str, default='./data/simulated',
                       help='Output directory for generated files')
    parser.add_argument('--num-tracks', type=int, default=10,
                       help='Number of tracks to simulate')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Simulation duration in seconds')
    parser.add_argument('--update-rate', type=float, default=10.0,
                       help='Update rate in Hz')
    parser.add_argument('--format', type=str, default='both',
                       choices=['binary', 'csv', 'both'],
                       help='Output format')
    parser.add_argument('--real-time', action='store_true',
                       help='Run in real-time mode')
    
    args = parser.parse_args()
    
    # Create scenario
    scenario = SimulationScenario(
        num_tracks=args.num_tracks,
        duration_seconds=args.duration,
        update_rate_hz=args.update_rate,
        output_format=args.format,
        real_time_mode=args.real_time,
    )
    
    # Run simulation
    simulator = TrackSimulator(scenario)
    output_files = simulator.run_simulation(args.output_dir)
    
    print(f"\nSimulation complete! Generated files:")
    for filepath in output_files:
        print(f"  {filepath}")


if __name__ == '__main__':
    main()
