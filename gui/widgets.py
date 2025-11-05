"""
Custom widgets for GUI application.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem
from PySide6.QtCore import Qt
import pyqtgraph as pg
import pandas as pd
import numpy as np
from typing import Dict, Optional


class TrackPlotWidget(QWidget):
    """Widget for visualizing track data"""
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        
        # Create plot widgets
        self.range_plot = pg.PlotWidget(title="Range vs Time")
        self.range_plot.setLabel('left', 'Range', units='m')
        self.range_plot.setLabel('bottom', 'Time', units='s')
        
        self.velocity_plot = pg.PlotWidget(title="Velocity vs Time")
        self.velocity_plot.setLabel('left', 'Speed', units='m/s')
        self.velocity_plot.setLabel('bottom', 'Time', units='s')
        
        self.height_plot = pg.PlotWidget(title="Height vs Time")
        self.height_plot.setLabel('left', 'Height', units='m')
        self.height_plot.setLabel('bottom', 'Time', units='s')
        
        self.trajectory_plot = pg.PlotWidget(title="2D Trajectory (Top View)")
        self.trajectory_plot.setLabel('left', 'Y Position', units='m')
        self.trajectory_plot.setLabel('bottom', 'X Position', units='m')
        self.trajectory_plot.setAspectLocked(True)
        
        layout.addWidget(self.range_plot)
        layout.addWidget(self.velocity_plot)
        layout.addWidget(self.height_plot)
        layout.addWidget(self.trajectory_plot)
    
    def plot_track(self, track_data: pd.DataFrame):
        """
        Plot track data.
        
        Args:
            track_data: DataFrame with track measurements
        """
        # Clear previous plots
        self.range_plot.clear()
        self.velocity_plot.clear()
        self.height_plot.clear()
        self.trajectory_plot.clear()
        
        if track_data.empty:
            return
        
        # Sort by timestamp
        track_data = track_data.sort_values('timestamp')
        
        # Normalize time to start at 0
        if 'timestamp' in track_data.columns:
            time = (track_data['timestamp'] - track_data['timestamp'].min()).values
        else:
            time = np.arange(len(track_data))
        
        # Plot range
        if 'range' in track_data.columns:
            self.range_plot.plot(time, track_data['range'].values, 
                               pen=pg.mkPen('b', width=2))
        elif all(col in track_data.columns for col in ['pos_x', 'pos_y', 'pos_z']):
            ranges = np.sqrt(track_data['pos_x']**2 + 
                           track_data['pos_y']**2 + 
                           track_data['pos_z']**2)
            self.range_plot.plot(time, ranges.values, 
                               pen=pg.mkPen('b', width=2))
        
        # Plot velocity
        if all(col in track_data.columns for col in ['vel_x', 'vel_y', 'vel_z']):
            speeds = np.sqrt(track_data['vel_x']**2 + 
                           track_data['vel_y']**2 + 
                           track_data['vel_z']**2)
            self.velocity_plot.plot(time, speeds.values, 
                                  pen=pg.mkPen('g', width=2))
        elif 'range_rate' in track_data.columns:
            # Use range rate as proxy
            self.velocity_plot.plot(time, track_data['range_rate'].values,
                                  pen=pg.mkPen('g', width=2))
        
        # Plot height
        if 'pos_z' in track_data.columns:
            self.height_plot.plot(time, track_data['pos_z'].values,
                                pen=pg.mkPen('r', width=2))
        
        # Plot 2D trajectory
        if 'pos_x' in track_data.columns and 'pos_y' in track_data.columns:
            self.trajectory_plot.plot(track_data['pos_x'].values,
                                     track_data['pos_y'].values,
                                     pen=pg.mkPen('m', width=2))
            
            # Mark start and end
            self.trajectory_plot.plot([track_data['pos_x'].iloc[0]],
                                     [track_data['pos_y'].iloc[0]],
                                     pen=None, symbol='o', symbolBrush='g', symbolSize=10)
            self.trajectory_plot.plot([track_data['pos_x'].iloc[-1]],
                                     [track_data['pos_y'].iloc[-1]],
                                     pen=None, symbol='s', symbolBrush='r', symbolSize=10)


class ModelMetricsWidget(QWidget):
    """Widget for displaying model metrics"""
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        
        self.info_label = QLabel("No models loaded")
        layout.addWidget(self.info_label)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels([
            "Model", "Tag", "Accuracy", "Precision", "F1 Score"
        ])
        
        layout.addWidget(self.metrics_table)
        
        # Feature importance plot
        self.importance_plot = pg.PlotWidget(title="Feature Importance")
        self.importance_plot.setLabel('left', 'Importance')
        self.importance_plot.setLabel('bottom', 'Feature Index')
        
        layout.addWidget(self.importance_plot)
    
    def set_models(self, models: Dict):
        """
        Set loaded models and display info.
        
        Args:
            models: Dictionary of loaded models
        """
        model_names = ", ".join(models.keys())
        self.info_label.setText(f"Loaded models: {model_names}")
    
    def display_metrics(self, metrics_df: pd.DataFrame):
        """
        Display model metrics.
        
        Args:
            metrics_df: DataFrame with model metrics
        """
        self.metrics_table.setRowCount(len(metrics_df))
        
        for row, (_, data) in enumerate(metrics_df.iterrows()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(data.get('model', ''))))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(str(data.get('tag', ''))))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{data.get('test_accuracy', 0):.3f}"))
            self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{data.get('precision', 0):.3f}"))
            self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{data.get('f1_score', 0):.3f}"))
        
        self.metrics_table.resizeColumnsToContents()
    
    def plot_feature_importance(self, importance: Dict[str, float]):
        """
        Plot feature importance.
        
        Args:
            importance: Dictionary of feature importances
        """
        self.importance_plot.clear()
        
        if not importance:
            return
        
        features = list(importance.keys())
        values = list(importance.values())
        
        # Create bar graph
        x = np.arange(len(features))
        bargraph = pg.BarGraphItem(x=x, height=values, width=0.8, brush='b')
        self.importance_plot.addItem(bargraph)
        
        # Set x-axis labels (simplified)
        self.importance_plot.getAxis('bottom').setTicks([
            [(i, features[i][:10]) for i in range(len(features))]
        ])
