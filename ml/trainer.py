"""Model training pipeline."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from pathlib import Path

from feature_store.feature_store import TrackFeatures
from .models import BaseModel


class ModelTrainer:
    """Handles model training and evaluation."""
    
    # Define all possible tags
    TAG_DEFINITIONS = {
        'high_speed': 'Speed > 300 m/s',
        'low_speed': 'Speed < 100 m/s',
        'high_maneuver': 'High acceleration variance',
        'linear_track': 'Low maneuver index',
        'climb': 'Positive altitude change',
        'descent': 'Negative altitude change',
        'two_jet': 'Likely two-jet engine (RCS pattern)',
        'multiengine': 'Multi-engine aircraft',
        'hover_like': 'Low speed + altitude maintenance',
        'unknown_engine': 'Cannot determine engine type',
    }
    
    def __init__(self):
        self.feature_names = [
            'flight_time', 'max_speed', 'min_speed', 'mean_speed', 'std_speed',
            'max_height', 'min_height', 'max_range', 'min_range',
            'maneuver_index', 'snr_mean', 'rcs_mean', 'doppler_mean'
        ]
    
    def prepare_data(self, tracks: List[TrackFeatures]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare training data from track features.
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Dict of tag_name -> labels (n_samples,)
        """
        X_list = []
        
        for track in tracks:
            features = track.get_feature_vector()
            X_list.append(features)
        
        X = np.array(X_list)
        
        # Generate labels (in real scenario, these would come from ground truth)
        y = self._generate_labels(tracks)
        
        return X, y
    
    def _generate_labels(self, tracks: List[TrackFeatures]) -> Dict[str, np.ndarray]:
        """
        Generate labels based on heuristics (for demo purposes).
        In production, use ground-truth labels.
        """
        n_tracks = len(tracks)
        
        labels = {
            'high_speed': np.zeros(n_tracks, dtype=int),
            'low_speed': np.zeros(n_tracks, dtype=int),
            'high_maneuver': np.zeros(n_tracks, dtype=int),
            'linear_track': np.zeros(n_tracks, dtype=int),
            'climb': np.zeros(n_tracks, dtype=int),
            'descent': np.zeros(n_tracks, dtype=int),
            'two_jet': np.zeros(n_tracks, dtype=int),
            'multiengine': np.zeros(n_tracks, dtype=int),
            'hover_like': np.zeros(n_tracks, dtype=int),
            'unknown_engine': np.zeros(n_tracks, dtype=int),
        }
        
        for i, track in enumerate(tracks):
            # High speed: mean speed > 300 m/s
            if track.mean_speed > 300:
                labels['high_speed'][i] = 1
            
            # Low speed: mean speed < 100 m/s
            if track.mean_speed < 100:
                labels['low_speed'][i] = 1
            
            # High maneuver: maneuver index > threshold
            if track.maneuver_index > 5.0:
                labels['high_maneuver'][i] = 1
            
            # Linear track: low maneuver index
            if track.maneuver_index < 2.0:
                labels['linear_track'][i] = 1
            
            # Climb: positive altitude change
            if len(track.positions) > 1:
                alt_change = track.positions[-1][2] - track.positions[0][2]
                if alt_change > 100:
                    labels['climb'][i] = 1
                elif alt_change < -100:
                    labels['descent'][i] = 1
            
            # Hover-like: low speed + small altitude variance
            if track.mean_speed < 50 and (track.max_height - track.min_height) < 50:
                labels['hover_like'][i] = 1
            
            # Engine type based on RCS patterns (simplified heuristics)
            if track.rcs_mean > 10:
                labels['multiengine'][i] = 1
            elif 5 < track.rcs_mean <= 10:
                labels['two_jet'][i] = 1
            else:
                labels['unknown_engine'][i] = 1
        
        return labels
    
    def train_model(self, model: BaseModel, tracks: List[TrackFeatures],
                   test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train model and return evaluation metrics.
        
        Returns:
            Dict with metrics: accuracy, precision, recall, f1 per tag
        """
        # Prepare data
        X, y = self.prepare_data(tracks)
        
        # Split data
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = {tag: labels[train_idx] for tag, labels in y.items()}
        y_test = {tag: labels[test_idx] for tag, labels in y.items()}
        
        # Train model
        print(f"Training {model.model_name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        metrics = self._compute_metrics(y_test, predictions)
        
        return metrics
    
    def _compute_metrics(self, y_true: Dict[str, np.ndarray], 
                        y_pred: Dict[str, np.ndarray]) -> Dict:
        """Compute evaluation metrics."""
        metrics = {}
        
        for tag_name in y_true.keys():
            true_labels = y_true[tag_name]
            pred_probs = y_pred[tag_name]
            pred_labels = (pred_probs > 0.5).astype(int)
            
            # Compute metrics
            accuracy = accuracy_score(true_labels, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='binary', zero_division=0
            )
            
            metrics[tag_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            }
        
        # Overall metrics
        all_true = np.concatenate([y_true[tag] for tag in y_true.keys()])
        all_pred = np.concatenate([(y_pred[tag] > 0.5).astype(int) for tag in y_pred.keys()])
        
        metrics['overall'] = {
            'accuracy': float(accuracy_score(all_true, all_pred)),
        }
        
        return metrics
    
    def cross_validate(self, model: BaseModel, tracks: List[TrackFeatures],
                      n_folds: int = 5) -> Dict:
        """Perform k-fold cross-validation."""
        from sklearn.model_selection import KFold
        
        X, y = self.prepare_data(tracks)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_metrics = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold_idx + 1}/{n_folds}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = {tag: labels[train_idx] for tag, labels in y.items()}
            y_test = {tag: labels[test_idx] for tag, labels in y.items()}
            
            # Train and evaluate
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = self._compute_metrics(y_test, predictions)
            
            fold_metrics.append(metrics)
        
        # Average metrics across folds
        avg_metrics = self._average_metrics(fold_metrics)
        return avg_metrics
    
    def _average_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Average metrics across folds."""
        avg_metrics = {}
        
        # Get all tags
        tags = list(fold_metrics[0].keys())
        
        for tag in tags:
            avg_metrics[tag] = {}
            metric_names = list(fold_metrics[0][tag].keys())
            
            for metric_name in metric_names:
                values = [fm[tag][metric_name] for fm in fold_metrics]
                avg_metrics[tag][metric_name] = float(np.mean(values))
                avg_metrics[tag][f'{metric_name}_std'] = float(np.std(values))
        
        return avg_metrics
    
    def save_metrics(self, metrics: Dict, output_path: str):
        """Save metrics to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_ground_truth_labels(self, filepath: str) -> Dict[int, Dict[str, int]]:
        """
        Load ground truth labels from file.
        
        Format: JSON with track_id -> {tag_name: label}
        """
        with open(filepath, 'r') as f:
            return json.load(f)
