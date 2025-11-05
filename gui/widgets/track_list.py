"""Track list widget."""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
                               QHeaderView, QLabel)
from PySide6.QtCore import Signal, Qt
from typing import List

from feature_store.feature_store import TrackFeatures


class TrackListWidget(QWidget):
    """Widget displaying list of tracks."""
    
    track_selected = Signal(int)
    
    def __init__(self):
        super().__init__()
        self.tracks = []
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout(self)
        
        label = QLabel("Tracks")
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(label)
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            'Track ID', 'Points', 'Avg Speed', 'Max Height', 'Flight Time', 'Tags'
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.table)
    
    def set_tracks(self, tracks: List[TrackFeatures]):
        """Set tracks to display."""
        self.tracks = tracks
        self.table.setRowCount(len(tracks))
        
        for i, track in enumerate(tracks):
            # Track ID
            self.table.setItem(i, 0, QTableWidgetItem(str(track.track_id)))
            
            # Number of points
            self.table.setItem(i, 1, QTableWidgetItem(str(len(track.timestamps))))
            
            # Average speed
            self.table.setItem(i, 2, QTableWidgetItem(f"{track.mean_speed:.1f} m/s"))
            
            # Max height
            self.table.setItem(i, 3, QTableWidgetItem(f"{track.max_height:.1f} m"))
            
            # Flight time
            self.table.setItem(i, 4, QTableWidgetItem(f"{track.flight_time:.1f} s"))
            
            # Tags
            if track.tags:
                tag_str = ", ".join([f"{k}" for k, v in track.tags.items() if v > 0.5])
            else:
                tag_str = "No tags"
            self.table.setItem(i, 5, QTableWidgetItem(tag_str))
    
    def on_selection_changed(self):
        """Handle selection change."""
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.tracks):
                track_id = self.tracks[row].track_id
                self.track_selected.emit(track_id)
