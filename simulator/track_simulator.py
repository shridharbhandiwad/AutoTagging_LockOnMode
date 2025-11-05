"""
Synthetic track data generator for testing and training.
Generates realistic aircraft track scenarios with various behavior patterns.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path


@dataclass
class SimulationConfig:
    """Configuration for track simulation"""
    duration: float = 60.0  # seconds
    dt: float = 0.1  # time step
    noise_level: float = 0.1  # measurement noise factor
    dropout_rate: float = 0.05  # probability of missed detection
    false_alarm_rate: float = 0.02  # probability of false alarm
    
    # Aircraft parameters
    initial_height: float = 5000.0  # meters
    initial_range: float = 10000.0  # meters
    base_speed: float = 200.0  # m/s
    
    # Sensor parameters
    range_noise: float = 10.0  # meters
    angle_noise: float = 0.01  # radians
    snr_mean: float = 20.0  # dB
    rcs_mean: float = 10.0  # dBsm


class TrackBehavior:
    """Base class for track behavior models"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def generate_trajectory(self) -> Dict[str, np.ndarray]:
        """
        Generate trajectory for this behavior.
        
        Returns:
            Dictionary with time, position, velocity, acceleration arrays
        """
        raise NotImplementedError


class LinearTrack(TrackBehavior):
    """Linear constant-velocity track"""
    
    def generate_trajectory(self) -> Dict[str, np.ndarray]:
        t = np.arange(0, self.config.duration, self.config.dt)
        n = len(t)
        
        # Initial position
        x0 = self.config.initial_range * np.cos(0.3)
        y0 = self.config.initial_range * np.sin(0.3)
        z0 = self.config.initial_height
        
        # Constant velocity
        vx = -self.config.base_speed * 0.8
        vy = -self.config.base_speed * 0.4
        vz = 0.0
        
        # Position (constant velocity)
        x = x0 + vx * t
        y = y0 + vy * t
        z = z0 + vz * t
        
        # Velocity (constant)
        vel_x = np.full(n, vx)
        vel_y = np.full(n, vy)
        vel_z = np.full(n, vz)
        
        # Acceleration (zero)
        acc_x = np.zeros(n)
        acc_y = np.zeros(n)
        acc_z = np.zeros(n)
        
        return {
            'time': t,
            'pos_x': x, 'pos_y': y, 'pos_z': z,
            'vel_x': vel_x, 'vel_y': vy, 'vel_z': vel_z,
            'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z
        }


class HighSpeedTrack(TrackBehavior):
    """High-speed track (fast aircraft)"""
    
    def generate_trajectory(self) -> Dict[str, np.ndarray]:
        t = np.arange(0, self.config.duration, self.config.dt)
        n = len(t)
        
        x0 = self.config.initial_range * np.cos(0.2)
        y0 = self.config.initial_range * np.sin(0.2)
        z0 = self.config.initial_height
        
        # High constant velocity
        vx = -self.config.base_speed * 2.5
        vy = -self.config.base_speed * 1.5
        vz = 20.0  # Slight climb
        
        x = x0 + vx * t
        y = y0 + vy * t
        z = z0 + vz * t
        
        vel_x = np.full(n, vx)
        vel_y = np.full(n, vy)
        vel_z = np.full(n, vz)
        
        acc_x = np.zeros(n)
        acc_y = np.zeros(n)
        acc_z = np.zeros(n)
        
        return {
            'time': t,
            'pos_x': x, 'pos_y': y, 'pos_z': z,
            'vel_x': vel_x, 'vel_y': vel_y, 'vel_z': vel_z,
            'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z
        }


class ManeuveringTrack(TrackBehavior):
    """High-maneuver track with turns and altitude changes"""
    
    def generate_trajectory(self) -> Dict[str, np.ndarray]:
        t = np.arange(0, self.config.duration, self.config.dt)
        n = len(t)
        
        x0 = self.config.initial_range * np.cos(0.4)
        y0 = self.config.initial_range * np.sin(0.4)
        z0 = self.config.initial_height
        
        # Circular maneuver
        omega = 0.1  # angular velocity (rad/s)
        radius = 2000.0  # turn radius (m)
        
        angle = omega * t
        
        x = x0 + radius * (np.cos(angle) - 1)
        y = y0 + radius * np.sin(angle)
        
        # Altitude variation (climb then descent)
        z = z0 + 500 * np.sin(2 * np.pi * t / self.config.duration)
        
        # Velocity from position derivative
        vel_x = -radius * omega * np.sin(angle)
        vel_y = radius * omega * np.cos(angle)
        vel_z = 500 * (2 * np.pi / self.config.duration) * np.cos(2 * np.pi * t / self.config.duration)
        
        # Acceleration from velocity derivative
        acc_x = -radius * omega**2 * np.cos(angle)
        acc_y = -radius * omega**2 * np.sin(angle)
        acc_z = -500 * (2 * np.pi / self.config.duration)**2 * np.sin(2 * np.pi * t / self.config.duration)
        
        return {
            'time': t,
            'pos_x': x, 'pos_y': y, 'pos_z': z,
            'vel_x': vel_x, 'vel_y': vel_y, 'vel_z': vel_z,
            'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z
        }


class ClimbingTrack(TrackBehavior):
    """Track with significant climb rate"""
    
    def generate_trajectory(self) -> Dict[str, np.ndarray]:
        t = np.arange(0, self.config.duration, self.config.dt)
        n = len(t)
        
        x0 = self.config.initial_range * 0.8
        y0 = self.config.initial_range * 0.3
        z0 = 1000.0  # Start lower
        
        vx = -self.config.base_speed * 0.6
        vy = -self.config.base_speed * 0.3
        vz = 50.0  # Strong climb
        
        x = x0 + vx * t
        y = y0 + vy * t
        z = z0 + vz * t
        
        vel_x = np.full(n, vx)
        vel_y = np.full(n, vy)
        vel_z = np.full(n, vz)
        
        acc_x = np.zeros(n)
        acc_y = np.zeros(n)
        acc_z = np.zeros(n)
        
        return {
            'time': t,
            'pos_x': x, 'pos_y': y, 'pos_z': z,
            'vel_x': vel_x, 'vel_y': vel_y, 'vel_z': vel_z,
            'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z
        }


class TrackSimulator:
    """Main simulator for generating synthetic track data"""
    
    BEHAVIOR_MAP = {
        'linear': LinearTrack,
        'high_speed': HighSpeedTrack,
        'maneuvering': ManeuveringTrack,
        'climb': ClimbingTrack,
    }
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
    
    def generate_track(self, behavior: str, track_id: int) -> pd.DataFrame:
        """
        Generate a single track.
        
        Args:
            behavior: Behavior type
            track_id: Track identifier
            
        Returns:
            DataFrame with track measurements
        """
        if behavior not in self.BEHAVIOR_MAP:
            raise ValueError(f"Unknown behavior: {behavior}")
        
        # Generate clean trajectory
        behavior_model = self.BEHAVIOR_MAP[behavior](self.config)
        trajectory = behavior_model.generate_trajectory()
        
        # Add measurements
        measurements = self._generate_measurements(trajectory, track_id, behavior)
        
        return measurements
    
    def _generate_measurements(self, trajectory: Dict[str, np.ndarray],
                              track_id: int, behavior: str) -> pd.DataFrame:
        """Generate noisy measurements from clean trajectory"""
        n = len(trajectory['time'])
        
        # Cartesian to spherical
        x = trajectory['pos_x']
        y = trajectory['pos_y']
        z = trajectory['pos_z']
        
        range_true = np.sqrt(x**2 + y**2 + z**2)
        azimuth_true = np.arctan2(y, x)
        elevation_true = np.arctan2(z, np.sqrt(x**2 + y**2))
        
        # Add noise
        range_meas = range_true + np.random.normal(0, self.config.range_noise, n)
        azimuth_meas = azimuth_true + np.random.normal(0, self.config.angle_noise, n)
        elevation_meas = elevation_true + np.random.normal(0, self.config.angle_noise, n)
        
        # Range rate (radial velocity)
        vx = trajectory['vel_x']
        vy = trajectory['vel_y']
        vz = trajectory['vel_z']
        
        range_rate = (x * vx + y * vy + z * vz) / range_true
        range_rate += np.random.normal(0, 1.0, n)
        
        # Signal parameters
        # RCS varies by aircraft type
        rcs_base = self.config.rcs_mean
        if behavior == 'high_speed':
            rcs_base += 5.0  # Larger aircraft
        elif behavior == 'two_jet':
            rcs_base += 3.0
        
        snr = np.random.normal(self.config.snr_mean, 3.0, n)
        rcs = np.random.normal(rcs_base, 2.0, n)
        
        # Doppler (simplified)
        doppler = -range_rate * 2 / 0.03  # Assuming X-band radar (wavelength ~3cm)
        
        # Apply dropouts
        valid_mask = np.random.random(n) > self.config.dropout_rate
        
        # Position errors
        pos_error_x = np.random.normal(0, 10.0, n)
        pos_error_y = np.random.normal(0, 10.0, n)
        pos_error_z = np.random.normal(0, 5.0, n)
        
        # Velocity errors
        vel_error_x = np.random.normal(0, 2.0, n)
        vel_error_y = np.random.normal(0, 2.0, n)
        vel_error_z = np.random.normal(0, 1.0, n)
        
        # Create dataframe
        data = {
            'track_id': track_id,
            'timestamp': trajectory['time'],
            'range': range_meas,
            'azimuth': azimuth_meas,
            'elevation': elevation_meas,
            'range_rate': range_rate,
            'snr': snr,
            'rcs': rcs,
            'doppler': doppler,
            'pos_x': trajectory['pos_x'],
            'pos_y': trajectory['pos_y'],
            'pos_z': trajectory['pos_z'],
            'vel_x': trajectory['vel_x'],
            'vel_y': trajectory['vel_y'],
            'vel_z': trajectory['vel_z'],
            'acc_x': trajectory['acc_x'],
            'acc_y': trajectory['acc_y'],
            'acc_z': trajectory['acc_z'],
            'pos_error_x': pos_error_x,
            'pos_error_y': pos_error_y,
            'pos_error_z': pos_error_z,
            'vel_error_x': vel_error_x,
            'vel_error_y': vel_error_y,
            'vel_error_z': vel_error_z,
            'valid': valid_mask,
        }
        
        df = pd.DataFrame(data)
        
        # Filter out dropouts
        df = df[df['valid']].drop(columns=['valid'])
        
        return df
    
    def generate_scenario(self, 
                         num_tracks: int = 10,
                         behavior_distribution: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Generate a complete scenario with multiple tracks.
        
        Args:
            num_tracks: Number of tracks to generate
            behavior_distribution: Distribution of behaviors (uniform if None)
            
        Returns:
            DataFrame with all tracks
        """
        if behavior_distribution is None:
            # Uniform distribution
            behaviors = list(self.BEHAVIOR_MAP.keys())
        else:
            # Sample according to distribution
            behaviors = list(behavior_distribution.keys())
            probs = list(behavior_distribution.values())
            probs = np.array(probs) / sum(probs)
        
        all_tracks = []
        
        for i in range(num_tracks):
            if behavior_distribution is None:
                behavior = behaviors[i % len(behaviors)]
            else:
                behavior = np.random.choice(behaviors, p=probs)
            
            track_data = self.generate_track(behavior, track_id=i+1)
            all_tracks.append(track_data)
        
        return pd.concat(all_tracks, ignore_index=True)
    
    def generate_ground_truth_tags(self, scenario_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ground truth tags for tracks.
        
        Args:
            scenario_df: DataFrame with track data
            
        Returns:
            DataFrame with ground truth tags
        """
        tags = []
        
        for track_id in scenario_df['track_id'].unique():
            track_data = scenario_df[scenario_df['track_id'] == track_id]
            
            # Compute statistics
            speeds = np.sqrt(track_data['vel_x']**2 + 
                           track_data['vel_y']**2 + 
                           track_data['vel_z']**2)
            
            accels = np.sqrt(track_data['acc_x']**2 + 
                           track_data['acc_y']**2 + 
                           track_data['acc_z']**2)
            
            mean_speed = speeds.mean()
            mean_accel = accels.mean()
            mean_vz = track_data['vel_z'].mean()
            
            # Assign tags based on behavior
            tag = {
                'track_id': track_id,
                'high_speed': mean_speed > 400,
                'low_speed': mean_speed < 150,
                'high_maneuver': mean_accel > 5.0,
                'linear_track': mean_accel < 2.0,
                'climb': mean_vz > 20,
                'descent': mean_vz < -20,
                'two_jet': track_data['rcs'].mean() > 12,
            }
            
            tags.append(tag)
        
        return pd.DataFrame(tags)
    
    def save_scenario(self, scenario_df: pd.DataFrame, 
                     output_dir: str,
                     format: str = 'binary'):
        """
        Save scenario to file.
        
        Args:
            scenario_df: Scenario dataframe
            output_dir: Output directory
            format: Format (binary/text/csv)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            filepath = output_path / "scenario.csv"
            scenario_df.to_csv(filepath, index=False)
        
        elif format == 'binary':
            # Save as binary using struct format
            from parsers.binary.struct_parser import TrackRecordParser
            filepath = output_path / "scenario.bin"
            
            # Convert to binary format
            # Simplified: save as numpy binary
            np.save(filepath, scenario_df.to_records(index=False))
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Saved scenario to {filepath}")
        
        return str(filepath)
