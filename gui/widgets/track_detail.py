"""Track detail widget."""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QScrollArea, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np

from feature_store.feature_store import TrackFeatures


class TrackDetailWidget(QWidget):
    """Widget showing detailed track information."""
    
    def __init__(self):
        super().__init__()
        self.current_track_id = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout(self)
        
        # Track info header
        self.header_label = QLabel("Select a track to view details")
        self.header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.header_label)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Summary group
        self.summary_group = QGroupBox("Track Summary")
        summary_layout = QGridLayout()
        self.summary_labels = {}
        
        summary_fields = [
            ('Track ID', 'track_id'),
            ('Flight Time', 'flight_time'),
            ('Max Speed', 'max_speed'),
            ('Mean Speed', 'mean_speed'),
            ('Max Height', 'max_height'),
            ('Max Range', 'max_range'),
            ('Maneuver Index', 'maneuver_index'),
            ('SNR Mean', 'snr_mean'),
            ('RCS Mean', 'rcs_mean'),
        ]
        
        for i, (label, key) in enumerate(summary_fields):
            row = i // 2
            col = (i % 2) * 2
            
            summary_layout.addWidget(QLabel(f"{label}:"), row, col)
            value_label = QLabel("-")
            self.summary_labels[key] = value_label
            summary_layout.addWidget(value_label, row, col + 1)
        
        self.summary_group.setLayout(summary_layout)
        self.scroll_layout.addWidget(self.summary_group)
        
        # Tags group
        self.tags_group = QGroupBox("Behavior Tags")
        self.tags_layout = QVBoxLayout()
        self.tags_group.setLayout(self.tags_layout)
        self.scroll_layout.addWidget(self.tags_group)
        
        # Plots
        self.setup_plots()
    
    def setup_plots(self):
        """Setup plot widgets."""
        # Range vs Time
        self.range_plot = pg.PlotWidget(title="Range vs Time")
        self.range_plot.setLabel('left', 'Range', units='m')
        self.range_plot.setLabel('bottom', 'Time', units='s')
        self.range_plot.showGrid(x=True, y=True)
        self.scroll_layout.addWidget(self.range_plot)
        
        # Velocity vs Time
        self.velocity_plot = pg.PlotWidget(title="Velocity vs Time")
        self.velocity_plot.setLabel('left', 'Velocity', units='m/s')
        self.velocity_plot.setLabel('bottom', 'Time', units='s')
        self.velocity_plot.showGrid(x=True, y=True)
        self.scroll_layout.addWidget(self.velocity_plot)
        
        # Height vs Time
        self.height_plot = pg.PlotWidget(title="Height vs Time")
        self.height_plot.setLabel('left', 'Height', units='m')
        self.height_plot.setLabel('bottom', 'Time', units='s')
        self.height_plot.showGrid(x=True, y=True)
        self.scroll_layout.addWidget(self.height_plot)
    
    def set_track(self, track: TrackFeatures):
        """Display track details."""
        self.current_track_id = track.track_id
        self.header_label.setText(f"Track {track.track_id}")
        
        # Update summary
        self.summary_labels['track_id'].setText(str(track.track_id))
        self.summary_labels['flight_time'].setText(f"{track.flight_time:.2f} s")
        self.summary_labels['max_speed'].setText(f"{track.max_speed:.2f} m/s")
        self.summary_labels['mean_speed'].setText(f"{track.mean_speed:.2f} m/s")
        self.summary_labels['max_height'].setText(f"{track.max_height:.2f} m")
        self.summary_labels['max_range'].setText(f"{track.max_range:.2f} m")
        self.summary_labels['maneuver_index'].setText(f"{track.maneuver_index:.4f}")
        self.summary_labels['snr_mean'].setText(f"{track.snr_mean:.2f} dB")
        self.summary_labels['rcs_mean'].setText(f"{track.rcs_mean:.2f}")
        
        # Update tags
        # Clear previous tags
        for i in reversed(range(self.tags_layout.count())):
            self.tags_layout.itemAt(i).widget().setParent(None)
        
        if track.tags:
            for tag_name, confidence in track.tags.items():
                tag_label = QLabel(f"{tag_name}: {confidence:.2%}")
                self.tags_layout.addWidget(tag_label)
        else:
            self.tags_layout.addWidget(QLabel("No tags assigned"))
        
        # Update plots
        self.update_plots(track)
    
    def update_plots(self, track: TrackFeatures):
        """Update plot data."""
        # Normalize timestamps to start at 0
        if track.timestamps:
            times = np.array(track.timestamps)
            times = (times - times[0]) / 1e6 if times[0] > 1e6 else times - times[0]
            
            # Range plot
            if track.ranges:
                self.range_plot.clear()
                self.range_plot.plot(times, track.ranges, pen='b', symbol='o', symbolSize=3)
            
            # Velocity plot
            if track.velocities:
                velocities = [np.linalg.norm(v) if len(v) == 3 else 0 for v in track.velocities]
                self.velocity_plot.clear()
                self.velocity_plot.plot(times[:len(velocities)], velocities, 
                                       pen='g', symbol='o', symbolSize=3)
            
            # Height plot
            if track.positions:
                heights = [p[2] if len(p) > 2 else 0 for p in track.positions]
                self.height_plot.clear()
                self.height_plot.plot(times[:len(heights)], heights,
                                     pen='r', symbol='o', symbolSize=3)
