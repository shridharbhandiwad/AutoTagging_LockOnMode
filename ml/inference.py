"""Model inference pipeline."""

import numpy as np
from typing import Dict, List, Optional
import time

from feature_store.feature_store import TrackFeatures
from .models import BaseModel


class ModelInference:
    """Handles model inference and multi-model ensemble."""
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.model_weights: Dict[str, float] = {}
    
    def add_model(self, name: str, model: BaseModel, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = model
        self.model_weights[name] = weight
    
    def predict_single_track(self, track: TrackFeatures, 
                            use_ensemble: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Predict tags for a single track.
        
        Returns:
            Dict of model_name -> {tag_name: confidence_score}
        """
        # Get feature vector
        features = track.get_feature_vector().reshape(1, -1)
        
        results = {}
        
        for model_name, model in self.models.items():
            if not model.is_fitted:
                continue
            
            start_time = time.time()
            predictions = model.predict(features)
            inference_time = time.time() - start_time
            
            # Convert to dict with float values
            tag_scores = {
                tag: float(pred[0]) for tag, pred in predictions.items()
            }
            
            results[model_name] = {
                'tags': tag_scores,
                'inference_time_ms': inference_time * 1000,
            }
        
        # Add ensemble prediction if requested
        if use_ensemble and len(results) > 1:
            ensemble_tags = self._ensemble_predictions(results)
            results['ensemble'] = {
                'tags': ensemble_tags,
                'inference_time_ms': sum(r['inference_time_ms'] for r in results.values())
            }
        
        return results
    
    def predict_batch(self, tracks: List[TrackFeatures],
                     model_name: Optional[str] = None) -> List[Dict[str, float]]:
        """
        Predict tags for batch of tracks using a specific model.
        
        Returns:
            List of {tag_name: confidence_score} dicts
        """
        if model_name is None:
            model_name = list(self.models.keys())[0]
        
        model = self.models[model_name]
        
        # Prepare features
        features = np.array([track.get_feature_vector() for track in tracks])
        
        # Predict
        predictions = model.predict(features)
        
        # Convert to list of dicts
        results = []
        for i in range(len(tracks)):
            tag_scores = {
                tag: float(pred[i]) for tag, pred in predictions.items()
            }
            results.append(tag_scores)
        
        return results
    
    def _ensemble_predictions(self, model_results: Dict) -> Dict[str, float]:
        """Combine predictions from multiple models using weighted voting."""
        # Exclude ensemble key if present
        model_names = [k for k in model_results.keys() if k != 'ensemble']
        
        if not model_names:
            return {}
        
        # Get all tags
        first_model = model_names[0]
        tags = list(model_results[first_model]['tags'].keys())
        
        # Weighted average
        ensemble_tags = {}
        total_weight = sum(self.model_weights.get(name, 1.0) for name in model_names)
        
        for tag in tags:
            weighted_sum = sum(
                model_results[name]['tags'][tag] * self.model_weights.get(name, 1.0)
                for name in model_names
            )
            ensemble_tags[tag] = weighted_sum / total_weight
        
        return ensemble_tags
    
    def apply_tags_to_track(self, track: TrackFeatures, 
                           model_name: str = 'ensemble',
                           threshold: float = 0.5) -> TrackFeatures:
        """
        Apply predicted tags to track object.
        
        Returns:
            Updated TrackFeatures object
        """
        results = self.predict_single_track(track, use_ensemble=True)
        
        if model_name not in results:
            model_name = list(results.keys())[0]
        
        tag_scores = results[model_name]['tags']
        
        # Apply threshold and update track
        track.tags = {
            tag: score for tag, score in tag_scores.items()
            if score > threshold
        }
        
        return track
    
    def get_model_comparison(self, tracks: List[TrackFeatures]) -> Dict:
        """
        Compare performance of all models on given tracks.
        
        Returns:
            Dict with per-model metrics
        """
        comparison = {}
        
        for model_name, model in self.models.items():
            if not model.is_fitted:
                continue
            
            # Time inference
            start_time = time.time()
            _ = self.predict_batch(tracks, model_name)
            total_time = time.time() - start_time
            
            comparison[model_name] = {
                'avg_inference_time_ms': (total_time / len(tracks)) * 1000,
                'total_time_s': total_time,
            }
        
        return comparison
