"""Main application window."""

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QFileDialog, QTabWidget,
                               QSplitter, QMessageBox, QProgressBar, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal, QMimeData
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from pathlib import Path
import traceback

from .widgets.file_browser import FileBrowserWidget
from .widgets.track_list import TrackListWidget
from .widgets.track_detail import TrackDetailWidget
from .widgets.model_manager import ModelManagerWidget
from .widgets.simulator_control import SimulatorControlWidget
from .processing_thread import ProcessingThread


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Airborne Track Behavior Tagger")
        self.resize(1400, 900)
        
        # Data
        self.current_tracks = {}
        self.feature_store = None
        self.model_inference = None
        
        # Setup UI
        self.setup_ui()
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def setup_ui(self):
        """Setup user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top toolbar
        toolbar_layout = QHBoxLayout()
        
        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.clicked.connect(self.open_file_dialog)
        toolbar_layout.addWidget(self.open_file_btn)
        
        self.run_inference_btn = QPushButton("Run Inference")
        self.run_inference_btn.clicked.connect(self.run_inference)
        self.run_inference_btn.setEnabled(False)
        toolbar_layout.addWidget(self.run_inference_btn)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        toolbar_layout.addWidget(self.export_btn)
        
        toolbar_layout.addStretch()
        
        self.status_label = QLabel("Ready")
        toolbar_layout.addWidget(self.status_label)
        
        main_layout.addLayout(toolbar_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Track Analysis Tab
        track_tab = QWidget()
        track_layout = QHBoxLayout(track_tab)
        
        # Splitter for track list and detail
        splitter = QSplitter(Qt.Horizontal)
        
        self.track_list_widget = TrackListWidget()
        self.track_list_widget.track_selected.connect(self.on_track_selected)
        splitter.addWidget(self.track_list_widget)
        
        self.track_detail_widget = TrackDetailWidget()
        splitter.addWidget(self.track_detail_widget)
        
        splitter.setSizes([400, 1000])
        track_layout.addWidget(splitter)
        
        self.tab_widget.addTab(track_tab, "Track Analysis")
        
        # Model Manager Tab
        self.model_manager_widget = ModelManagerWidget()
        self.model_manager_widget.model_loaded.connect(self.on_model_loaded)
        self.tab_widget.addTab(self.model_manager_widget, "Model Manager")
        
        # Simulator Tab
        self.simulator_widget = SimulatorControlWidget()
        self.simulator_widget.simulation_complete.connect(self.on_simulation_complete)
        self.tab_widget.addTab(self.simulator_widget, "Simulator")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            self.load_file(filepath)
    
    def open_file_dialog(self):
        """Open file selection dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Track Data File",
            "",
            "All Files (*);;Binary Files (*.bin *.dat);;Text Files (*.csv *.txt *.log)"
        )
        
        if filepath:
            self.load_file(filepath)
    
    def load_file(self, filepath: str):
        """Load and process file."""
        self.status_label.setText(f"Loading: {Path(filepath).name}")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Create processing thread
        self.processing_thread = ProcessingThread(filepath)
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished_signal.connect(self.on_processing_complete)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()
    
    def on_progress(self, message: str):
        """Update progress message."""
        self.status_label.setText(message)
        self.statusBar().showMessage(message)
    
    def on_processing_complete(self, tracks: dict, feature_store):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Loaded {len(tracks)} tracks")
        self.statusBar().showMessage(f"Successfully loaded {len(tracks)} tracks")
        
        self.current_tracks = tracks
        self.feature_store = feature_store
        
        # Update track list
        self.track_list_widget.set_tracks(list(tracks.values()))
        
        # Enable buttons
        self.run_inference_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
    
    def on_processing_error(self, error_msg: str):
        """Handle processing error."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error")
        
        QMessageBox.critical(self, "Error", f"Failed to process file:\n{error_msg}")
    
    def on_track_selected(self, track_id: int):
        """Handle track selection."""
        if track_id in self.current_tracks:
            track = self.current_tracks[track_id]
            self.track_detail_widget.set_track(track)
    
    def on_model_loaded(self, model_inference):
        """Handle model loaded event."""
        self.model_inference = model_inference
        self.run_inference_btn.setEnabled(len(self.current_tracks) > 0)
        self.statusBar().showMessage("Model loaded successfully")
    
    def run_inference(self):
        """Run inference on loaded tracks."""
        if not self.model_inference:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
        
        if not self.current_tracks:
            QMessageBox.warning(self, "Warning", "No tracks loaded")
            return
        
        self.status_label.setText("Running inference...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.current_tracks))
        
        try:
            # Run inference on all tracks
            for i, (track_id, track) in enumerate(self.current_tracks.items()):
                self.model_inference.apply_tags_to_track(track, model_name='ensemble')
                self.progress_bar.setValue(i + 1)
            
            # Update display
            self.track_list_widget.set_tracks(list(self.current_tracks.values()))
            
            # Refresh current track detail if one is selected
            current_track_id = self.track_detail_widget.current_track_id
            if current_track_id and current_track_id in self.current_tracks:
                self.track_detail_widget.set_track(self.current_tracks[current_track_id])
            
            self.progress_bar.setVisible(False)
            self.status_label.setText("Inference complete")
            self.statusBar().showMessage("Inference complete")
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Inference failed:\n{str(e)}")
    
    def export_results(self):
        """Export results to file."""
        if not self.current_tracks:
            QMessageBox.warning(self, "Warning", "No tracks to export")
            return
        
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "track_results.csv",
            "CSV Files (*.csv);;Parquet Files (*.parquet)"
        )
        
        if filepath:
            try:
                format = 'parquet' if filepath.endswith('.parquet') else 'csv'
                self.feature_store.export_all_tracks(filepath, format=format)
                
                QMessageBox.information(self, "Success", f"Results exported to:\n{filepath}")
                self.statusBar().showMessage(f"Exported to {Path(filepath).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
    
    def on_simulation_complete(self, output_files: list):
        """Handle simulation completion."""
        if output_files:
            # Ask if user wants to load simulated data
            reply = QMessageBox.question(
                self,
                "Simulation Complete",
                f"Simulation generated {len(output_files)} files.\nLoad the simulated data?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Load first data file (binary or csv)
                for filepath in output_files:
                    if filepath.endswith(('.bin', '.csv')):
                        self.load_file(filepath)
                        break
