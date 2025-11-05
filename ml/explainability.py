"""Explainability and feature importance analysis."""

import numpy as np
import shap
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from .models import BaseModel, RandomForestTagger, XGBoostTagger


class ExplainabilityAnalyzer:
    """Provides explainability for model predictions."""
    
    def __init__(self, model: BaseModel, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importances from tree-based models.
        
        Returns:
            Dict of tag_name -> {feature_name: importance}
        """
        if isinstance(self.model, (RandomForestTagger, XGBoostTagger)):
            importances = self.model.get_feature_importance()
            
            result = {}
            for tag_name, importance_values in importances.items():
                result[tag_name] = {
                    fname: float(imp)
                    for fname, imp in zip(self.feature_names, importance_values)
                }
            
            return result
        else:
            return {}
    
    def compute_shap_values(self, X: np.ndarray, tag_name: Optional[str] = None,
                           max_samples: int = 100) -> Dict:
        """
        Compute SHAP values for interpretability.
        
        Args:
            X: Feature matrix
            tag_name: Specific tag to analyze (if None, analyze all)
            max_samples: Max samples for SHAP (for performance)
        
        Returns:
            Dict with SHAP values and base values
        """
        if not isinstance(self.model, (RandomForestTagger, XGBoostTagger)):
            return {}
        
        # Limit samples for performance
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Scale features
        X_scaled = self.model.scaler.transform(X_sample)
        
        shap_results = {}
        
        # Select tags to analyze
        tags_to_analyze = [tag_name] if tag_name else self.model.tag_names
        
        for tag in tags_to_analyze:
            if tag not in self.model.models:
                continue
            
            model = self.model.models[tag]
            
            try:
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class
                
                shap_results[tag] = {
                    'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                    'base_value': float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value),
                    'feature_names': self.feature_names,
                }
            except Exception as e:
                print(f"Warning: Could not compute SHAP values for {tag}: {e}")
                continue
        
        return shap_results
    
    def plot_feature_importance(self, tag_name: str, output_path: str,
                               top_n: int = 10):
        """
        Plot feature importance for a specific tag.
        
        Args:
            tag_name: Tag to visualize
            output_path: Path to save plot
            top_n: Number of top features to show
        """
        importances = self.get_feature_importance()
        
        if tag_name not in importances:
            print(f"No importance data for tag: {tag_name}")
            return
        
        # Get top N features
        tag_importances = importances[tag_name]
        sorted_features = sorted(tag_importances.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        features, values = zip(*sorted_features)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance for {tag_name}')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_shap_summary(self, X: np.ndarray, tag_name: str, 
                         output_path: str, max_samples: int = 100):
        """
        Plot SHAP summary for a specific tag.
        
        Args:
            X: Feature matrix
            tag_name: Tag to visualize
            output_path: Path to save plot
            max_samples: Max samples for visualization
        """
        shap_results = self.compute_shap_values(X, tag_name, max_samples)
        
        if tag_name not in shap_results:
            print(f"No SHAP values for tag: {tag_name}")
            return
        
        shap_values = np.array(shap_results[tag_name]['shap_values'])
        
        # Limit samples for visualization
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
            shap_values = shap_values[:max_samples]
        else:
            X_sample = X
        
        X_scaled = self.model.scaler.transform(X_sample)
        
        # Create SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_scaled, 
                         feature_names=self.feature_names,
                         show=False)
        plt.title(f'SHAP Summary for {tag_name}')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def explain_prediction(self, features: np.ndarray, tag_name: str) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            features: Feature vector (1D array)
            tag_name: Tag to explain
        
        Returns:
            Dict with explanation
        """
        if tag_name not in self.model.models:
            return {}
        
        # Reshape for prediction
        X = features.reshape(1, -1)
        X_scaled = self.model.scaler.transform(X)
        
        # Get prediction
        model = self.model.models[tag_name]
        pred_proba = model.predict_proba(X_scaled)[0]
        prediction = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
        
        # Get feature contributions (simplified)
        importances = self.get_feature_importance()
        
        if tag_name in importances:
            feature_contributions = {
                fname: float(features[i] * importances[tag_name].get(fname, 0))
                for i, fname in enumerate(self.feature_names)
            }
        else:
            feature_contributions = {}
        
        return {
            'prediction': float(prediction),
            'feature_values': {fname: float(features[i]) 
                             for i, fname in enumerate(self.feature_names)},
            'feature_contributions': feature_contributions,
        }
