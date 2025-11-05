"""
Machine learning package for track behavior tagging.
"""
from ml.feature_store import FeatureStore, TrackFeatures, TrackTags, TrackMeasurement, ProcessedTrackState
from ml.models.classical_models import ClassicalTrackTagger
from ml.models.deep_models import DeepTrackTagger
from ml.training.pipeline import TrainingPipeline, InferencePipeline
from ml.inference.explainability import ExplainabilityAnalyzer

__all__ = [
    'FeatureStore',
    'TrackFeatures',
    'TrackTags',
    'TrackMeasurement',
    'ProcessedTrackState',
    'ClassicalTrackTagger',
    'DeepTrackTagger',
    'TrainingPipeline',
    'InferencePipeline',
    'ExplainabilityAnalyzer',
]
