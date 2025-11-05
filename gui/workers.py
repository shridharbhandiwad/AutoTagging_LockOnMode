"""
Worker threads for background processing.
"""
from PySide6.QtCore import QThread, Signal
import pandas as pd
from typing import Optional

from parsers import FileRouter


class FileProcessWorker(QThread):
    """Worker thread for file parsing"""
    
    finished = Signal(pd.DataFrame)
    error = Signal(str)
    progress = Signal(int)
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        """Run file processing"""
        try:
            # Parse file
            data = FileRouter.parse_file(self.file_path)
            
            # Validate
            if FileRouter.validate_track_data(data):
                self.finished.emit(data)
            else:
                self.error.emit("Invalid track data: missing required fields")
        
        except Exception as e:
            self.error.emit(str(e))


class InferenceWorker(QThread):
    """Worker thread for model inference"""
    
    finished = Signal(pd.DataFrame)
    error = Signal(str)
    progress = Signal(int)
    
    def __init__(self, inference_pipeline, features_df: pd.DataFrame):
        super().__init__()
        self.inference_pipeline = inference_pipeline
        self.features_df = features_df
    
    def run(self):
        """Run inference"""
        try:
            # Run inference
            predictions = self.inference_pipeline.predict(self.features_df)
            
            self.finished.emit(predictions)
        
        except Exception as e:
            self.error.emit(str(e))


class FeatureExtractionWorker(QThread):
    """Worker thread for feature extraction"""
    
    finished = Signal(pd.DataFrame)
    error = Signal(str)
    progress = Signal(int)
    
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data
    
    def run(self):
        """Run feature extraction"""
        try:
            # TODO: Implement feature extraction from measurements
            # For now, just pass through
            self.finished.emit(self.data)
        
        except Exception as e:
            self.error.emit(str(e))
