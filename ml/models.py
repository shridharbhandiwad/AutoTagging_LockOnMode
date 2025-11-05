"""ML/DL models for track behavior tagging."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path


class BaseModel:
    """Base class for all models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Train model on features and labels."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict labels for features."""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save model to disk."""
        raise NotImplementedError
    
    def load(self, path: str):
        """Load model from disk."""
        raise NotImplementedError


class RandomForestTagger(BaseModel):
    """Random Forest multi-label classifier for track tagging."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models: Dict[str, RandomForestClassifier] = {}
        self.tag_names: List[str] = []
    
    def fit(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Train separate RF for each tag."""
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        self.tag_names = list(y.keys())
        
        for tag_name, tag_labels in y.items():
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled, tag_labels)
            self.models[tag_name] = model
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict probabilities for each tag."""
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for tag_name, model in self.models.items():
            probs = model.predict_proba(X_scaled)
            # Get probability of positive class
            predictions[tag_name] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importances for each tag."""
        importances = {}
        for tag_name, model in self.models.items():
            importances[tag_name] = model.feature_importances_
        return importances
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'random_forest.pkl', 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'tag_names': self.tag_names,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
            }, f)
    
    def load(self, path: str):
        """Load model from disk."""
        with open(Path(path) / 'random_forest.pkl', 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scaler = data['scaler']
            self.tag_names = data['tag_names']
            self.n_estimators = data['n_estimators']
            self.max_depth = data['max_depth']
            self.is_fitted = True


class XGBoostTagger(BaseModel):
    """XGBoost multi-label classifier for track tagging."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models: Dict[str, xgb.XGBClassifier] = {}
        self.tag_names: List[str] = []
    
    def fit(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Train separate XGBoost for each tag."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.tag_names = list(y.keys())
        
        for tag_name, tag_labels in y.items():
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            model.fit(X_scaled, tag_labels)
            self.models[tag_name] = model
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict probabilities for each tag."""
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for tag_name, model in self.models.items():
            probs = model.predict_proba(X_scaled)
            predictions[tag_name] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importances for each tag."""
        importances = {}
        for tag_name, model in self.models.items():
            importances[tag_name] = model.feature_importances_
        return importances
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'xgboost.pkl', 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'tag_names': self.tag_names,
            }, f)
    
    def load(self, path: str):
        """Load model from disk."""
        with open(Path(path) / 'xgboost.pkl', 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scaler = data['scaler']
            self.tag_names = data['tag_names']
            self.is_fitted = True


class LSTMTagger(BaseModel):
    """LSTM-based sequence model for track tagging."""
    
    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, num_layers: int = 2, 
                 num_tags: int = 10, dropout: float = 0.3):
        super().__init__("LSTM")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tags = num_tags
        self.dropout = dropout
        
        self.model = LSTMNet(input_dim, hidden_dim, num_layers, num_tags, dropout)
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.tag_names: List[str] = []
    
    def fit(self, X_seq: np.ndarray, y: Dict[str, np.ndarray], 
            epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        """
        Train LSTM on sequences.
        
        Args:
            X_seq: Shape (n_samples, seq_len, n_features)
            y: Dict of tag_name -> labels (n_samples,)
        """
        self.tag_names = list(y.keys())
        
        # Convert labels to multi-label format
        y_multi = np.column_stack([y[tag] for tag in self.tag_names])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_multi).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = (len(X_tensor) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_tensor))
                
                X_batch = X_tensor[start_idx:end_idx]
                y_batch = y_tensor[start_idx:end_idx]
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/num_batches:.4f}")
        
        self.is_fitted = True
    
    def predict(self, X_seq: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict probabilities for sequences.
        
        Args:
            X_seq: Shape (n_samples, seq_len, n_features)
        
        Returns:
            Dict of tag_name -> probabilities
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        predictions = {}
        for i, tag_name in enumerate(self.tag_names):
            predictions[tag_name] = probs[:, i]
        
        return predictions
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state': self.model.state_dict(),
            'tag_names': self.tag_names,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_tags': self.num_tags,
            'dropout': self.dropout,
        }, path / 'lstm.pth')
    
    def load(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(Path(path) / 'lstm.pth', map_location=self.device)
        
        self.tag_names = checkpoint['tag_names']
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.num_tags = checkpoint['num_tags']
        self.dropout = checkpoint['dropout']
        
        self.model = LSTMNet(
            self.input_dim, self.hidden_dim, self.num_layers, 
            self.num_tags, self.dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.is_fitted = True


class LSTMNet(nn.Module):
    """LSTM network architecture."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 num_tags: int, dropout: float = 0.3):
        super(LSTMNet, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_tags)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # Shape: (batch, hidden_dim)
        
        # Apply dropout and fc
        out = self.dropout(last_hidden)
        out = self.fc(out)  # Shape: (batch, num_tags)
        
        return out
