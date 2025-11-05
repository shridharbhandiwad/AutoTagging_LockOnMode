"""Model manager widget."""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QComboBox, QTableWidget, QTableWidgetItem,
                               QFileDialog, QMessageBox, QGroupBox)
from PySide6.QtCore import Signal
from pathlib import Path

from ml.models import RandomForestTagger, XGBoostTagger, LSTMTagger
from ml.inference import ModelInference


class ModelManagerWidget(QWidget):
    """Widget for managing ML models."""
    
    model_loaded = Signal(object)  # ModelInference object
    
    def __init__(self):
        super().__init__()
        self.model_inference = ModelInference()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout(self)
        
        # Load model section
        load_group = QGroupBox("Load Model")
        load_layout = QVBoxLayout()
        
        load_btn_layout = QHBoxLayout()
        
        self.load_rf_btn = QPushButton("Load Random Forest")
        self.load_rf_btn.clicked.connect(lambda: self.load_model('RandomForest'))
        load_btn_layout.addWidget(self.load_rf_btn)
        
        self.load_xgb_btn = QPushButton("Load XGBoost")
        self.load_xgb_btn.clicked.connect(lambda: self.load_model('XGBoost'))
        load_btn_layout.addWidget(self.load_xgb_btn)
        
        self.load_lstm_btn = QPushButton("Load LSTM")
        self.load_lstm_btn.clicked.connect(lambda: self.load_model('LSTM'))
        load_btn_layout.addWidget(self.load_lstm_btn)
        
        load_layout.addLayout(load_btn_layout)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Loaded models section
        models_group = QGroupBox("Loaded Models")
        models_layout = QVBoxLayout()
        
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(3)
        self.models_table.setHorizontalHeaderLabels(['Model Name', 'Type', 'Status'])
        models_layout.addWidget(self.models_table)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # Model info
        info_label = QLabel("Load pre-trained models to run inference on tracks")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
    
    def load_model(self, model_type: str):
        """Load a model from file."""
        if model_type == 'RandomForest':
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select Random Forest Model Directory",
                "./models/saved"
            )
            
            if dir_path and Path(dir_path).exists():
                try:
                    model = RandomForestTagger()
                    model.load(dir_path)
                    self.model_inference.add_model('RandomForest', model, weight=1.0)
                    self.update_models_table()
                    self.model_loaded.emit(self.model_inference)
                    QMessageBox.information(self, "Success", "Random Forest model loaded")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
        
        elif model_type == 'XGBoost':
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select XGBoost Model Directory",
                "./models/saved"
            )
            
            if dir_path and Path(dir_path).exists():
                try:
                    model = XGBoostTagger()
                    model.load(dir_path)
                    self.model_inference.add_model('XGBoost', model, weight=1.0)
                    self.update_models_table()
                    self.model_loaded.emit(self.model_inference)
                    QMessageBox.information(self, "Success", "XGBoost model loaded")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
        
        elif model_type == 'LSTM':
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select LSTM Model Directory",
                "./models/saved"
            )
            
            if dir_path and Path(dir_path).exists():
                try:
                    model = LSTMTagger()
                    model.load(dir_path)
                    self.model_inference.add_model('LSTM', model, weight=1.0)
                    self.update_models_table()
                    self.model_loaded.emit(self.model_inference)
                    QMessageBox.information(self, "Success", "LSTM model loaded")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
    
    def update_models_table(self):
        """Update the models table."""
        models = self.model_inference.models
        self.models_table.setRowCount(len(models))
        
        for i, (name, model) in enumerate(models.items()):
            self.models_table.setItem(i, 0, QTableWidgetItem(name))
            self.models_table.setItem(i, 1, QTableWidgetItem(model.model_name))
            status = "Loaded" if model.is_fitted else "Not trained"
            self.models_table.setItem(i, 2, QTableWidgetItem(status))
