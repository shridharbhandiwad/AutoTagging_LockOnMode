"""
Training pipeline for track behavior tagging models.
Supports classical and deep learning models with cross-validation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from sklearn.model_selection import KFold
from ml.models.classical_models import ClassicalTrackTagger
from ml.models.deep_models import DeepTrackTagger
from ml.feature_store import FeatureStore, TrackFeatures, TrackTags


class TrainingPipeline:
    """Complete training pipeline for track taggers"""
    
    def __init__(self, feature_store: FeatureStore):
        """
        Initialize pipeline.
        
        Args:
            feature_store: Feature store instance
        """
        self.feature_store = feature_store
        self.models = {}
        self.metrics = {}
    
    def train_classical_models(self, 
                              features_df: pd.DataFrame,
                              tags_df: pd.DataFrame,
                              model_types: List[str] = ['RandomForest', 'XGBoost', 'LightGBM'],
                              test_size: float = 0.2,
                              cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Train classical models.
        
        Args:
            features_df: Features dataframe
            tags_df: Ground truth tags dataframe
            model_types: List of model types to train
            test_size: Test set fraction
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary of trained models and metrics
        """
        results = {}
        
        for model_type in model_types:
            print(f"\n=== Training {model_type} ===")
            
            # Create and train model
            tagger = ClassicalTrackTagger(model_name=model_type)
            metrics = tagger.train(features_df, tags_df, test_size=test_size)
            
            # Cross-validation
            if cv_folds > 1:
                cv_metrics = self._cross_validate_classical(
                    tagger, features_df, tags_df, cv_folds
                )
                metrics['cv_scores'] = cv_metrics
            
            # Store model and metrics
            self.models[model_type] = tagger
            self.metrics[model_type] = metrics
            
            results[model_type] = {
                'model': tagger,
                'metrics': metrics
            }
        
        return results
    
    def _cross_validate_classical(self, 
                                 tagger: ClassicalTrackTagger,
                                 features_df: pd.DataFrame,
                                 tags_df: pd.DataFrame,
                                 n_folds: int) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation"""
        data = features_df.merge(tags_df[['track_id'] + tagger.TAG_COLUMNS], 
                                on='track_id', how='inner')
        
        X = data[tagger.FEATURE_COLUMNS].fillna(0)
        
        cv_scores = {tag: [] for tag in tagger.TAG_COLUMNS}
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"  Fold {fold + 1}/{n_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            
            for tag in tagger.TAG_COLUMNS:
                if tag not in data.columns:
                    continue
                
                y = data[tag].astype(int)
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if len(y_train.unique()) < 2:
                    continue
                
                # Train model for this tag
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model = tagger._create_model(tag)
                model.fit(X_train_scaled, y_train)
                
                score = model.score(X_val_scaled, y_val)
                cv_scores[tag].append(score)
        
        # Compute mean scores
        mean_scores = {tag: np.mean(scores) if scores else 0.0 
                      for tag, scores in cv_scores.items()}
        
        return mean_scores
    
    def train_deep_models(self,
                         measurements_df: pd.DataFrame,
                         tags_df: pd.DataFrame,
                         model_types: List[str] = ['LSTM', 'Transformer'],
                         feature_columns: List[str] = None,
                         epochs: int = 50,
                         batch_size: int = 32) -> Dict[str, Dict]:
        """
        Train deep learning models.
        
        Args:
            measurements_df: Time-series measurements
            tags_df: Ground truth tags
            model_types: Model types to train
            feature_columns: Sequence features to use
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary of trained models and metrics
        """
        if feature_columns is None:
            feature_columns = [
                'range', 'azimuth', 'elevation', 'range_rate',
                'snr', 'rcs', 'doppler'
            ]
        
        results = {}
        
        for model_type in model_types:
            print(f"\n=== Training {model_type} ===")
            
            tagger = DeepTrackTagger(model_type=model_type)
            
            # Prepare sequences
            sequences, track_ids = tagger.prepare_sequences(
                measurements_df, feature_columns
            )
            
            # Prepare labels
            labels = []
            for track_id in track_ids:
                tag_row = tags_df[tags_df['track_id'] == track_id]
                if len(tag_row) > 0:
                    label_vec = tag_row[tagger.TAG_COLUMNS].values[0].astype(float)
                else:
                    label_vec = np.zeros(len(tagger.TAG_COLUMNS))
                labels.append(label_vec)
            
            # Train model
            history = tagger.train(
                sequences, labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2
            )
            
            self.models[model_type] = tagger
            self.metrics[model_type] = history
            
            results[model_type] = {
                'model': tagger,
                'history': history
            }
        
        return results
    
    def save_models(self, output_dir: str):
        """Save all trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_dir = output_path / model_name
            model_dir.mkdir(exist_ok=True)
            model.save(str(model_dir))
            
            print(f"Saved {model_name} to {model_dir}")
        
        # Save metrics
        metrics_file = output_path / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            # Convert numpy types to native Python types
            serializable_metrics = {}
            for model_name, metrics in self.metrics.items():
                serializable_metrics[model_name] = self._make_serializable(metrics)
            
            json.dump(serializable_metrics, f, indent=2)
    
    def _make_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def compare_models(self) -> pd.DataFrame:
        """Compare performance of all trained models"""
        comparison = []
        
        for model_name, metrics in self.metrics.items():
            if isinstance(metrics, dict) and 'high_speed' in metrics:
                # Classical model
                for tag, tag_metrics in metrics.items():
                    if isinstance(tag_metrics, dict) and 'test_accuracy' in tag_metrics:
                        comparison.append({
                            'model': model_name,
                            'tag': tag,
                            'test_accuracy': tag_metrics['test_accuracy'],
                            'f1_score': tag_metrics.get('f1_score', 0),
                            'precision': tag_metrics.get('precision', 0),
                            'recall': tag_metrics.get('recall', 0)
                        })
            elif isinstance(metrics, dict) and 'val_accuracy' in metrics:
                # Deep learning model
                final_val_acc = metrics['val_accuracy'][-1] if metrics['val_accuracy'] else 0
                comparison.append({
                    'model': model_name,
                    'tag': 'all',
                    'test_accuracy': final_val_acc,
                    'f1_score': 0,
                    'precision': 0,
                    'recall': 0
                })
        
        return pd.DataFrame(comparison)


class InferencePipeline:
    """Inference pipeline for track tagging"""
    
    def __init__(self):
        """Initialize pipeline"""
        self.models = {}
    
    def load_models(self, models_dir: str, 
                   model_names: Optional[List[str]] = None):
        """
        Load trained models.
        
        Args:
            models_dir: Directory containing saved models
            model_names: Specific models to load (load all if None)
        """
        models_path = Path(models_dir)
        
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Detect available models
        if model_names is None:
            model_names = [d.name for d in models_path.iterdir() if d.is_dir()]
        
        for model_name in model_names:
            model_dir = models_path / model_name
            
            if not model_dir.exists():
                print(f"Warning: Model directory not found: {model_dir}")
                continue
            
            # Try to load as classical or deep model
            if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                model = ClassicalTrackTagger(model_name=model_name)
                model.load(str(model_dir))
            elif model_name in ['LSTM', 'Transformer']:
                model = DeepTrackTagger(model_type=model_name)
                model.load(str(model_dir))
            else:
                print(f"Warning: Unknown model type: {model_name}")
                continue
            
            self.models[model_name] = model
            print(f"Loaded {model_name} from {model_dir}")
    
    def predict(self, 
               features_df: pd.DataFrame,
               model_name: Optional[str] = None,
               ensemble: bool = False) -> pd.DataFrame:
        """
        Run inference.
        
        Args:
            features_df: Features dataframe
            model_name: Specific model to use (use all if None)
            ensemble: Use ensemble prediction
            
        Returns:
            DataFrame with predictions
        """
        if not self.models:
            raise ValueError("No models loaded")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model not loaded: {model_name}")
            models_to_use = {model_name: self.models[model_name]}
        else:
            models_to_use = self.models
        
        predictions = {}
        
        for name, model in models_to_use.items():
            if isinstance(model, ClassicalTrackTagger):
                pred_df = model.predict(features_df)
            else:
                # For deep models, would need sequences
                # Simplified: use aggregated features
                print(f"Warning: Deep model {name} requires sequences, skipping")
                continue
            
            predictions[name] = pred_df
        
        if ensemble and len(predictions) > 1:
            # Ensemble: average confidences
            return self._ensemble_predictions(predictions, features_df)
        elif predictions:
            # Return first model's predictions
            return list(predictions.values())[0]
        else:
            return pd.DataFrame()
    
    def _ensemble_predictions(self, 
                             predictions: Dict[str, pd.DataFrame],
                             features_df: pd.DataFrame) -> pd.DataFrame:
        """Ensemble multiple model predictions"""
        result = {'track_id': features_df['track_id'].values}
        
        tag_columns = ClassicalTrackTagger.TAG_COLUMNS
        
        for tag in tag_columns:
            conf_column = f'{tag}_conf'
            
            # Average confidences across models
            confidences = []
            for pred_df in predictions.values():
                if conf_column in pred_df.columns:
                    confidences.append(pred_df[conf_column].values)
            
            if confidences:
                avg_conf = np.mean(confidences, axis=0)
                result[conf_column] = avg_conf
                result[tag] = avg_conf > 0.5
            else:
                result[conf_column] = 0.0
                result[tag] = False
        
        result['model_name'] = 'ensemble'
        
        return pd.DataFrame(result)
