"""Tests for simulator."""

import pytest
import tempfile
from pathlib import Path

from simulator.simulator import TrackSimulator, SimulationScenario


class TestSimulator:
    """Tests for track simulator."""
    
    def test_scenario_creation(self):
        """Test creating simulation scenario."""
        scenario = SimulationScenario(
            num_tracks=5,
            duration_seconds=10.0,
            update_rate_hz=10.0,
        )
        
        assert scenario.num_tracks == 5
        assert scenario.duration_seconds == 10.0
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        scenario = SimulationScenario(num_tracks=3)
        simulator = TrackSimulator(scenario)
        simulator.initialize_tracks()
        
        assert len(simulator.tracks) == 3
    
    def test_simulation_step(self):
        """Test single simulation step."""
        scenario = SimulationScenario(num_tracks=2)
        simulator = TrackSimulator(scenario)
        simulator.initialize_tracks()
        
        measurements = simulator.step()
        
        # Should generate some measurements (may have dropouts)
        assert isinstance(measurements, list)
    
    def test_full_simulation(self):
        """Test full simulation run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = SimulationScenario(
                num_tracks=3,
                duration_seconds=2.0,
                update_rate_hz=10.0,
                output_format='csv',
            )
            
            simulator = TrackSimulator(scenario)
            output_files = simulator.run_simulation(tmpdir)
            
            assert len(output_files) > 0
            
            # Check files exist
            for filepath in output_files:
                assert Path(filepath).exists()
    
    def test_binary_output(self):
        """Test binary output generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = SimulationScenario(
                num_tracks=2,
                duration_seconds=1.0,
                update_rate_hz=5.0,
                output_format='binary',
            )
            
            simulator = TrackSimulator(scenario)
            output_files = simulator.run_simulation(tmpdir)
            
            # Find binary file
            binary_files = [f for f in output_files if f.endswith('.bin')]
            assert len(binary_files) > 0
            
            # Check file has content
            binary_file = Path(binary_files[0])
            assert binary_file.stat().st_size > 0
