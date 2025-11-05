"""
Explainability module using SHAP and feature importance analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import shap
from ml.models.classical_models import ClassicalTrackTagger


class ExplainabilityAnalyzer:
    """Analyze and explain model predictions"""
    
    def __init__(self, model: ClassicalTrackTagger):
        """
        Initialize analyzer.
        
        Args:
            model: Trained classical model
        """
        self.model = model
        self.explainers = {}
        self.shap_values = {}
    
    def compute_feature_importance(self, tag: str, 
                                   top_n: int = 10) -> Dict[str, float]:
        """
        Get feature importance for a specific tag.
        
        Args:
            tag: Tag name
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature importances
        """
        return self.model.get_feature_importance(tag, top_n=top_n)
    
    def compute_shap_values(self, 
                           features_df: pd.DataFrame,
                           tag: str,
                           background_samples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for explanations.
        
        Args:
            features_df: Features dataframe
            tag: Tag to explain
            background_samples: Number of background samples for SHAP
            
        Returns:
            SHAP values array
        """
        if tag not in self.model.models:
            raise ValueError(f"Tag {tag} not found in model")
        
        model = self.model.models[tag]
        scaler = self.model.scalers[tag]
        
        X = features_df[self.model.FEATURE_COLUMNS].fillna(0)
        X_scaled = scaler.transform(X)
        
        # Sample background data
        if len(X_scaled) > background_samples:
            background = shap.sample(X_scaled, background_samples, random_state=42)
        else:
            background = X_scaled
        
        # Create SHAP explainer
        try:
            if hasattr(model, 'predict_proba'):
                # Tree-based explainer for tree models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)
                
                # For binary classification, take positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                # Kernel explainer for other models
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_scaled)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
        
        except Exception as e:
            print(f"Warning: SHAP computation failed for {tag}: {e}")
            # Fallback to simple feature importance
            if hasattr(model, 'feature_importances_'):
                shap_values = np.tile(model.feature_importances_, (len(X_scaled), 1))
            else:
                shap_values = np.zeros((len(X_scaled), len(self.model.FEATURE_COLUMNS)))
        
        self.explainers[tag] = explainer if 'explainer' in locals() else None
        self.shap_values[tag] = shap_values
        
        return shap_values
    
    def explain_prediction(self, 
                          track_id: int,
                          features_df: pd.DataFrame,
                          tag: str,
                          top_n: int = 5) -> Dict[str, float]:
        """
        Explain prediction for a specific track.
        
        Args:
            track_id: Track ID to explain
            features_df: Features dataframe
            tag: Tag to explain
            top_n: Number of top contributing features
            
        Returns:
            Dictionary of feature contributions
        """
        track_data = features_df[features_df['track_id'] == track_id]
        
        if track_data.empty:
            return {}
        
        # Compute SHAP values if not already computed
        if tag not in self.shap_values:
            self.compute_shap_values(features_df, tag)
        
        # Get track index
        track_idx = features_df[features_df['track_id'] == track_id].index[0]
        overall_idx = features_df.index.get_loc(track_idx)
        
        # Get SHAP values for this track
        track_shap = self.shap_values[tag][overall_idx]
        
        # Create feature contribution dictionary
        contributions = dict(zip(self.model.FEATURE_COLUMNS, track_shap))
        
        # Sort by absolute contribution
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return dict(sorted_contributions[:top_n])
    
    def plot_feature_importance(self, 
                               tag: str,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance for a tag.
        
        Args:
            tag: Tag name
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        importance = self.compute_feature_importance(tag, top_n=15)
        
        if not importance:
            print(f"No importance data for tag: {tag}")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(importance.keys())
        values = list(importance.values())
        
        ax.barh(features, values)
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance for "{tag}" tag')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_shap_summary(self,
                         features_df: pd.DataFrame,
                         tag: str,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot SHAP summary plot.
        
        Args:
            features_df: Features dataframe
            tag: Tag to plot
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if tag not in self.shap_values:
            self.compute_shap_values(features_df, tag)
        
        X = features_df[self.model.FEATURE_COLUMNS].fillna(0)
        shap_values = self.shap_values[tag]
        
        fig = plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=self.model.FEATURE_COLUMNS,
            show=False
        )
        
        plt.title(f'SHAP Summary for "{tag}" tag')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_explanation_report(self,
                                   track_id: int,
                                   features_df: pd.DataFrame,
                                   predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report for a track.
        
        Args:
            track_id: Track ID
            features_df: Features dataframe
            predictions_df: Predictions dataframe
            
        Returns:
            Explanation report dictionary
        """
        report = {
            'track_id': track_id,
            'predictions': {},
            'explanations': {}
        }
        
        # Get predictions
        pred_row = predictions_df[predictions_df['track_id'] == track_id]
        if pred_row.empty:
            return report
        
        # Get features
        feat_row = features_df[features_df['track_id'] == track_id]
        if feat_row.empty:
            return report
        
        # For each predicted tag
        for tag in self.model.TAG_COLUMNS:
            if tag in pred_row.columns:
                prediction = bool(pred_row[tag].values[0])
                confidence = float(pred_row[f'{tag}_conf'].values[0])
                
                report['predictions'][tag] = {
                    'predicted': prediction,
                    'confidence': confidence
                }
                
                # Explain prediction
                if prediction:  # Only explain positive predictions
                    contributions = self.explain_prediction(
                        track_id, features_df, tag, top_n=5
                    )
                    report['explanations'][tag] = contributions
        
        return report


class PermutationImportance:
    """Compute permutation importance for any model"""
    
    @staticmethod
    def compute(model, X: np.ndarray, y: np.ndarray, 
               feature_names: List[str],
               n_repeats: int = 10) -> Dict[str, Tuple[float, float]]:
        """
        Compute permutation importance.
        
        Args:
            model: Trained model
            X: Features array
            y: Labels array
            feature_names: Feature names
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary of feature importances (mean, std)
        """
        from sklearn.metrics import accuracy_score
        
        # Baseline score
        baseline_score = accuracy_score(y, model.predict(X))
        
        importances = {name: [] for name in feature_names}
        
        for i in range(len(feature_names)):
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                
                permuted_score = accuracy_score(y, model.predict(X_permuted))
                importance = baseline_score - permuted_score
                
                importances[feature_names[i]].append(importance)
        
        # Compute mean and std
        result = {}
        for name, values in importances.items():
            result[name] = (np.mean(values), np.std(values))
        
        return result
