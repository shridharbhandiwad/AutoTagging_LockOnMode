"""Track data simulator for testing."""

import numpy as np
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import time
import pandas as pd


@dataclass
class SimulationScenario:
    """Configuration for simulation scenario."""
    num_tracks: int = 5
    duration_seconds: float = 60.0
    update_rate_hz: float = 10.0
    
    # Track behavior distribution
    prob_high_speed: float = 0.3
    prob_low_speed: float = 0.2
    prob_high_maneuver: float = 0.25
    prob_linear: float = 0.4
    
    # Noise parameters
    position_noise_std: float = 5.0  # meters
    velocity_noise_std: float = 1.0  # m/s
    measurement_dropout_prob: float = 0.05
    false_alarm_rate: float = 0.02
    
    # Output settings
    output_format: str = 'binary'  # 'binary', 'csv', or 'both'
    real_time_mode: bool = False


class TrackSimulator:
    """Generates synthetic track data for testing."""
    
    def __init__(self, scenario: SimulationScenario):
        self.scenario = scenario
        self.tracks: List[Track] = []
        self.current_time = 0.0
        self.dt = 1.0 / scenario.update_rate_hz
    
    def initialize_tracks(self):
        """Create initial track states."""
        self.tracks = []
        
        for track_id in range(self.scenario.num_tracks):
            # Random initial conditions
            track_type = self._sample_track_type()
            
            track = Track(
                track_id=track_id,
                track_type=track_type,
                position=self._generate_initial_position(),
                velocity=self._generate_initial_velocity(track_type),
            )
            
            self.tracks.append(track)
    
    def _sample_track_type(self) -> str:
        """Sample track behavior type."""
        rand = np.random.rand()
        
        if rand < self.scenario.prob_high_speed:
            return 'high_speed'
        elif rand < self.scenario.prob_high_speed + self.scenario.prob_low_speed:
            return 'low_speed'
        elif rand < (self.scenario.prob_high_speed + self.scenario.prob_low_speed + 
                     self.scenario.prob_high_maneuver):
            return 'high_maneuver'
        else:
            return 'linear'
    
    def _generate_initial_position(self) -> np.ndarray:
        """Generate random initial position."""
        # Start in a box: x, y in [-5000, 5000], z in [1000, 10000]
        x = np.random.uniform(-5000, 5000)
        y = np.random.uniform(-5000, 5000)
        z = np.random.uniform(1000, 10000)
        return np.array([x, y, z])
    
    def _generate_initial_velocity(self, track_type: str) -> np.ndarray:
        """Generate initial velocity based on track type."""
        if track_type == 'high_speed':
            speed = np.random.uniform(300, 500)  # m/s
        elif track_type == 'low_speed':
            speed = np.random.uniform(50, 100)
        else:
            speed = np.random.uniform(150, 300)
        
        # Random direction
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(-np.pi/6, np.pi/6)  # Mostly horizontal
        
        vx = speed * np.cos(phi) * np.cos(theta)
        vy = speed * np.cos(phi) * np.sin(theta)
        vz = speed * np.sin(phi)
        
        return np.array([vx, vy, vz])
    
    def step(self) -> List[dict]:
        """Advance simulation by one time step."""
        measurements = []
        
        for track in self.tracks:
            # Update track dynamics
            self._update_track(track)
            
            # Generate measurement (with dropout)
            if np.random.rand() > self.scenario.measurement_dropout_prob:
                measurement = self._generate_measurement(track)
                measurements.append(measurement)
        
        # Add false alarms
        num_false_alarms = np.random.poisson(
            self.scenario.false_alarm_rate * len(self.tracks)
        )
        for _ in range(num_false_alarms):
            false_measurement = self._generate_false_alarm()
            measurements.append(false_measurement)
        
        self.current_time += self.dt
        
        return measurements
    
    def _update_track(self, track: Track):
        """Update track state based on dynamics."""
        # Add acceleration based on track type
        if track.track_type == 'high_maneuver':
            # Random acceleration
            accel = np.random.randn(3) * 20.0  # m/s^2
        elif track.track_type == 'linear':
            # Minimal acceleration
            accel = np.random.randn(3) * 1.0
        else:
            # Moderate acceleration
            accel = np.random.randn(3) * 5.0
        
        # Update velocity and position
        track.velocity += accel * self.dt
        track.position += track.velocity * self.dt
        track.acceleration = accel
        
        # Keep altitude reasonable
        if track.position[2] < 500:
            track.velocity[2] = abs(track.velocity[2])
        elif track.position[2] > 15000:
            track.velocity[2] = -abs(track.velocity[2])
    
    def _generate_measurement(self, track: Track) -> dict:
        """Generate noisy measurement from track."""
        # Add measurement noise
        pos_noise = np.random.randn(3) * self.scenario.position_noise_std
        vel_noise = np.random.randn(3) * self.scenario.velocity_noise_std
        
        noisy_position = track.position + pos_noise
        noisy_velocity = track.velocity + vel_noise
        
        # Convert to spherical coordinates
        range_val, azimuth, elevation = self._cartesian_to_spherical(noisy_position)
        range_rate = self._compute_range_rate(noisy_position, noisy_velocity)
        
        # Generate signal characteristics
        snr = self._compute_snr(range_val, track.track_type)
        rcs = self._compute_rcs(track.track_type)
        doppler = range_rate / 3e8 * 10e9  # Simplified doppler (10 GHz radar)
        
        measurement = {
            'timestamp': int(self.current_time * 1e6),  # microseconds
            'track_id': track.track_id,
            'range': range_val,
            'azimuth': azimuth,
            'elevation': elevation,
            'range_rate': range_rate,
            'snr': snr,
            'rcs': rcs,
            'doppler': doppler,
            'position': noisy_position.tolist(),
            'velocity': noisy_velocity.tolist(),
            'acceleration': track.acceleration.tolist(),
            'pos_error': pos_noise.tolist(),
            'vel_error': vel_noise.tolist(),
            'measurement_valid': 1,
            'i_sample': np.random.randn(),
            'q_sample': np.random.randn(),
            'sp_value': snr + np.random.randn() * 2,
        }
        
        return measurement
    
    def _generate_false_alarm(self) -> dict:
        """Generate false alarm measurement."""
        # Random position
        random_pos = self._generate_initial_position()
        range_val, azimuth, elevation = self._cartesian_to_spherical(random_pos)
        
        measurement = {
            'timestamp': int(self.current_time * 1e6),
            'track_id': 999999,  # Invalid track ID
            'range': range_val,
            'azimuth': azimuth,
            'elevation': elevation,
            'range_rate': np.random.randn() * 50,
            'snr': np.random.uniform(5, 15),  # Low SNR
            'rcs': np.random.uniform(0.1, 2),
            'doppler': np.random.randn() * 1000,
            'position': random_pos.tolist(),
            'velocity': [0, 0, 0],
            'acceleration': [0, 0, 0],
            'pos_error': [0, 0, 0],
            'vel_error': [0, 0, 0],
            'measurement_valid': 0,
            'i_sample': np.random.randn(),
            'q_sample': np.random.randn(),
            'sp_value': np.random.uniform(5, 15),
        }
        
        return measurement
    
    def _cartesian_to_spherical(self, position: np.ndarray) -> Tuple[float, float, float]:
        """Convert Cartesian to spherical coordinates."""
        x, y, z = position
        range_val = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(z / range_val) if range_val > 0 else 0
        return range_val, azimuth, elevation
    
    def _compute_range_rate(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """Compute range rate (radial velocity)."""
        range_val = np.linalg.norm(position)
        if range_val < 1e-6:
            return 0.0
        return np.dot(position, velocity) / range_val
    
    def _compute_snr(self, range_val: float, track_type: str) -> float:
        """Compute SNR based on range and track type."""
        # Simple range-dependent SNR model
        base_snr = 40 - 20 * np.log10(range_val / 1000)  # dB
        
        # Add noise
        snr = base_snr + np.random.randn() * 3
        
        return max(snr, 5)  # Minimum 5 dB
    
    def _compute_rcs(self, track_type: str) -> float:
        """Compute RCS based on track type."""
        # Different aircraft have different RCS
        if track_type == 'high_speed':
            rcs_mean = 15  # Large aircraft
        elif track_type == 'low_speed':
            rcs_mean = 5   # Small aircraft
        else:
            rcs_mean = 10
        
        rcs = rcs_mean + np.random.randn() * 2
        return max(rcs, 0.1)
    
    def run_simulation(self, output_dir: str) -> List[str]:
        """
        Run complete simulation and save outputs.
        
        Returns:
            List of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracks
        self.initialize_tracks()
        
        # Run simulation
        all_measurements = []
        num_steps = int(self.scenario.duration_seconds * self.scenario.update_rate_hz)
        
        print(f"Running simulation: {num_steps} steps, {len(self.tracks)} tracks")
        
        for step in range(num_steps):
            measurements = self.step()
            all_measurements.extend(measurements)
            
            if self.scenario.real_time_mode:
                time.sleep(self.dt)
            
            if (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{num_steps}")
        
        print(f"Generated {len(all_measurements)} measurements")
        
        # Save outputs
        output_files = []
        
        if self.scenario.output_format in ['binary', 'both']:
            binary_file = output_dir / 'simulated_tracks.bin'
            self._save_binary(all_measurements, binary_file)
            output_files.append(str(binary_file))
        
        if self.scenario.output_format in ['csv', 'both']:
            csv_file = output_dir / 'simulated_tracks.csv'
            self._save_csv(all_measurements, csv_file)
            output_files.append(str(csv_file))
        
        # Save ground truth labels
        labels_file = output_dir / 'ground_truth_labels.json'
        self._save_ground_truth(labels_file)
        output_files.append(str(labels_file))
        
        return output_files
    
    def _save_binary(self, measurements: List[dict], filepath: Path):
        """Save measurements to binary file."""
        with open(filepath, 'wb') as f:
            for m in measurements:
                # Pack according to TrackRecord structure
                record = struct.pack(
                    '<QIfffffff3f3f3f3f3fB',
                    m['timestamp'],
                    m['track_id'],
                    m['range'],
                    m['azimuth'],
                    m['elevation'],
                    m['range_rate'],
                    m['snr'],
                    m['rcs'],
                    m['doppler'],
                    *m['position'],
                    *m['velocity'],
                    *m['acceleration'],
                    *m['pos_error'],
                    *m['vel_error'],
                    m['measurement_valid']
                )
                f.write(record)
    
    def _save_csv(self, measurements: List[dict], filepath: Path):
        """Save measurements to CSV file."""
        # Flatten nested lists
        flat_measurements = []
        for m in measurements:
            flat_m = {
                'timestamp': m['timestamp'],
                'track_id': m['track_id'],
                'range': m['range'],
                'azimuth': m['azimuth'],
                'elevation': m['elevation'],
                'range_rate': m['range_rate'],
                'snr': m['snr'],
                'rcs': m['rcs'],
                'doppler': m['doppler'],
                'pos_x': m['position'][0],
                'pos_y': m['position'][1],
                'pos_z': m['position'][2],
                'vel_x': m['velocity'][0],
                'vel_y': m['velocity'][1],
                'vel_z': m['velocity'][2],
                'measurement_valid': m['measurement_valid'],
            }
            flat_measurements.append(flat_m)
        
        df = pd.DataFrame(flat_measurements)
        df.to_csv(filepath, index=False)
    
    def _save_ground_truth(self, filepath: Path):
        """Save ground truth labels for tracks."""
        import json
        
        labels = {}
        for track in self.tracks:
            track_labels = {
                'high_speed': 1 if track.track_type == 'high_speed' else 0,
                'low_speed': 1 if track.track_type == 'low_speed' else 0,
                'high_maneuver': 1 if track.track_type == 'high_maneuver' else 0,
                'linear_track': 1 if track.track_type == 'linear' else 0,
            }
            labels[track.track_id] = track_labels
        
        with open(filepath, 'w') as f:
            json.dump(labels, f, indent=2)


@dataclass
class Track:
    """Track state."""
    track_id: int
    track_type: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray = None
    
    def __post_init__(self):
        if self.acceleration is None:
            self.acceleration = np.zeros(3)
