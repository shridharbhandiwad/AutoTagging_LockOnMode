"""
Simulator package for generating synthetic track data.
"""
from simulator.track_simulator import (
    TrackSimulator,
    SimulationConfig,
    TrackBehavior,
    LinearTrack,
    HighSpeedTrack,
    ManeuveringTrack,
    ClimbingTrack
)

__all__ = [
    'TrackSimulator',
    'SimulationConfig',
    'TrackBehavior',
    'LinearTrack',
    'HighSpeedTrack',
    'ManeuveringTrack',
    'ClimbingTrack',
]
