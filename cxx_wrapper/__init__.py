"""
Python wrapper for C++ algorithm libraries.
Provides Kalman filtering, gating, and feature extraction.
"""

try:
    from . import cxxlib
    
    # Export main classes
    KalmanFilter = cxxlib.KalmanFilter
    FilterResult = cxxlib.FilterResult
    Gating = cxxlib.Gating
    FeatureExtractor = cxxlib.FeatureExtractor
    TrackFeatures = cxxlib.TrackFeatures
    
    __all__ = [
        'KalmanFilter',
        'FilterResult',
        'Gating',
        'FeatureExtractor',
        'TrackFeatures',
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import C++ extension module: {e}\n"
        "C++ algorithms will not be available. Run 'python setup.py build_ext --inplace' to build."
    )
    
    # Provide stub implementations for development
    class KalmanFilter:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("C++ module not built")
    
    class Gating:
        @staticmethod
        def mahalanobis_distance(*args, **kwargs):
            raise NotImplementedError("C++ module not built")
    
    class FeatureExtractor:
        @staticmethod
        def extract_features(*args, **kwargs):
            raise NotImplementedError("C++ module not built")
