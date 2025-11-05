"""
Main window for Airborne Track Tagger GUI application.
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QTabWidget, QFileDialog,
    QMessageBox, QProgressBar, QSplitter, QTextEdit, QComboBox, QGroupBox
)
from PySide6.QtCore import Qt, QThread, Signal, QMimeData
from PySide6.QtGui import QDragEnterEvent, QDropEvent
import pandas as pd
from pathlib import Path
from typing import Optional

from parsers import FileRouter
from ml import InferencePipeline, FeatureStore
from gui.widgets import TrackPlotWidget, ModelMetricsWidget
from gui.workers import FileProcessWorker, InferenceWorker


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Airborne Track Behavior Tagger")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.feature_store = FeatureStore()
        self.inference_pipeline = InferencePipeline()
        self.current_data = None
        self.current_features = None
        self.current_tags = None
        
        # Setup UI
        self.setup_ui()
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Top toolbar
        toolbar = self.create_toolbar()
        main_layout.addLayout(toolbar)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: File list and track table
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Visualizations and details
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
        # Bottom status bar
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(status_layout)
    
    def create_toolbar(self) -> QHBoxLayout:
        """Create top toolbar"""
        toolbar = QHBoxLayout()
        
        # File operations
        self.browse_btn = QPushButton("Browse Files")
        self.browse_btn.clicked.connect(self.browse_files)
        
        self.parse_btn = QPushButton("Parse File")
        self.parse_btn.clicked.connect(self.parse_file)
        self.parse_btn.setEnabled(False)
        
        # Model operations
        self.load_model_btn = QPushButton("Load Models")
        self.load_model_btn.clicked.connect(self.load_models)
        
        self.run_inference_btn = QPushButton("Run Inference")
        self.run_inference_btn.clicked.connect(self.run_inference)
        self.run_inference_btn.setEnabled(False)
        
        # Export
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        
        toolbar.addWidget(QLabel("File:"))
        toolbar.addWidget(self.browse_btn)
        toolbar.addWidget(self.parse_btn)
        toolbar.addSpacing(20)
        toolbar.addWidget(QLabel("Model:"))
        toolbar.addWidget(self.load_model_btn)
        toolbar.addWidget(self.run_inference_btn)
        toolbar.addSpacing(20)
        toolbar.addWidget(self.export_btn)
        toolbar.addStretch()
        
        return toolbar
    
    def create_left_panel(self) -> QWidget:
        """Create left panel with file info and track list"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File info group
        file_group = QGroupBox("File Information")
        file_layout = QVBoxLayout(file_group)
        
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setWordWrap(True)
        file_layout.addWidget(self.file_info_label)
        
        # Drag & drop area
        drop_area = QLabel("Drag & Drop Files Here")
        drop_area.setAlignment(Qt.AlignCenter)
        drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #999;
                border-radius: 5px;
                padding: 20px;
                background-color: #f5f5f5;
            }
        """)
        drop_area.setMinimumHeight(80)
        file_layout.addWidget(drop_area)
        
        layout.addWidget(file_group)
        
        # Track list table
        track_group = QGroupBox("Tracks")
        track_layout = QVBoxLayout(track_group)
        
        self.track_table = QTableWidget()
        self.track_table.setColumnCount(6)
        self.track_table.setHorizontalHeaderLabels([
            "Track ID", "Measurements", "Duration (s)", 
            "Mean Speed", "Mean Height", "Tags"
        ])
        self.track_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.track_table.itemSelectionChanged.connect(self.on_track_selected)
        
        track_layout.addWidget(self.track_table)
        
        layout.addWidget(track_group)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create right panel with visualizations"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # Track details tab
        self.track_plot_widget = TrackPlotWidget()
        self.tabs.addTab(self.track_plot_widget, "Track Visualization")
        
        # Model metrics tab
        self.metrics_widget = ModelMetricsWidget()
        self.tabs.addTab(self.metrics_widget, "Model Metrics")
        
        # Tags & explanations tab
        tags_widget = QWidget()
        tags_layout = QVBoxLayout(tags_widget)
        
        self.tags_table = QTableWidget()
        self.tags_table.setColumnCount(3)
        self.tags_table.setHorizontalHeaderLabels(["Tag", "Value", "Confidence"])
        tags_layout.addWidget(QLabel("Behavior Tags:"))
        tags_layout.addWidget(self.tags_table)
        
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMaximumHeight(200)
        tags_layout.addWidget(QLabel("Explanation:"))
        tags_layout.addWidget(self.explanation_text)
        
        self.tabs.addTab(tags_widget, "Tags & Explanations")
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            self.load_file(files[0])
    
    def browse_files(self):
        """Browse for files"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Track Data File",
            "",
            "All Files (*.*);;Binary Files (*.bin *.raw);;Text Files (*.txt *.csv)"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path: str):
        """Load file"""
        self.current_file = file_path
        self.file_info_label.setText(f"File: {Path(file_path).name}\nDetecting type...")
        
        # Detect file type
        file_type = FileRouter.detect_file_type(file_path)
        self.file_info_label.setText(
            f"File: {Path(file_path).name}\nType: {file_type}"
        )
        
        self.parse_btn.setEnabled(True)
    
    def parse_file(self):
        """Parse loaded file"""
        if not hasattr(self, 'current_file'):
            return
        
        self.status_label.setText("Parsing file...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Create worker thread
        self.parse_worker = FileProcessWorker(self.current_file)
        self.parse_worker.finished.connect(self.on_parse_complete)
        self.parse_worker.error.connect(self.on_parse_error)
        self.parse_worker.start()
    
    def on_parse_complete(self, data: pd.DataFrame):
        """Handle parse completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Parsed {len(data)} records")
        
        self.current_data = data
        self.populate_track_table()
        
        self.run_inference_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
    
    def on_parse_error(self, error_msg: str):
        """Handle parse error"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Parse failed")
        
        QMessageBox.critical(self, "Parse Error", f"Failed to parse file:\n{error_msg}")
    
    def populate_track_table(self):
        """Populate track list table"""
        if self.current_data is None:
            return
        
        # Group by track_id
        grouped = self.current_data.groupby('track_id')
        
        self.track_table.setRowCount(len(grouped))
        
        for row, (track_id, track_data) in enumerate(grouped):
            # Track ID
            self.track_table.setItem(row, 0, QTableWidgetItem(str(track_id)))
            
            # Number of measurements
            self.track_table.setItem(row, 1, QTableWidgetItem(str(len(track_data))))
            
            # Duration
            if 'timestamp' in track_data.columns:
                duration = track_data['timestamp'].max() - track_data['timestamp'].min()
                self.track_table.setItem(row, 2, QTableWidgetItem(f"{duration:.2f}"))
            
            # Mean speed (if velocity columns exist)
            if all(col in track_data.columns for col in ['vel_x', 'vel_y', 'vel_z']):
                speeds = (track_data['vel_x']**2 + track_data['vel_y']**2 + 
                         track_data['vel_z']**2)**0.5
                self.track_table.setItem(row, 3, QTableWidgetItem(f"{speeds.mean():.2f}"))
            
            # Mean height
            if 'pos_z' in track_data.columns:
                self.track_table.setItem(row, 4, QTableWidgetItem(
                    f"{track_data['pos_z'].mean():.2f}"))
            
            # Tags (will be filled after inference)
            self.track_table.setItem(row, 5, QTableWidgetItem(""))
        
        self.track_table.resizeColumnsToContents()
    
    def load_models(self):
        """Load trained models"""
        model_dir = QFileDialog.getExistingDirectory(
            self, "Select Models Directory", "./models"
        )
        
        if model_dir:
            try:
                self.inference_pipeline.load_models(model_dir)
                self.status_label.setText(
                    f"Loaded {len(self.inference_pipeline.models)} models"
                )
                
                # Update metrics widget
                self.metrics_widget.set_models(self.inference_pipeline.models)
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", 
                                   f"Failed to load models:\n{str(e)}")
    
    def run_inference(self):
        """Run inference on loaded data"""
        if self.current_data is None:
            return
        
        if not self.inference_pipeline.models:
            QMessageBox.warning(self, "No Models", 
                              "Please load models first")
            return
        
        self.status_label.setText("Running inference...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        # TODO: Extract features from current_data
        # For now, use a placeholder
        
        QMessageBox.information(self, "Inference", 
                               "Inference completed (placeholder)")
        
        self.progress_bar.setVisible(False)
        self.status_label.setText("Inference complete")
    
    def on_track_selected(self):
        """Handle track selection"""
        selected_rows = self.track_table.selectedItems()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        track_id = int(self.track_table.item(row, 0).text())
        
        # Get track data
        if self.current_data is not None:
            track_data = self.current_data[self.current_data['track_id'] == track_id]
            
            # Update visualization
            self.track_plot_widget.plot_track(track_data)
            
            # Update tags if available
            if self.current_tags is not None:
                self.update_tags_display(track_id)
    
    def update_tags_display(self, track_id: int):
        """Update tags display for selected track"""
        # TODO: Implement tags display
        pass
    
    def export_results(self):
        """Export results to CSV"""
        if self.current_data is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.current_data.to_csv(file_path, index=False)
                self.status_label.setText(f"Exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error",
                                   f"Failed to export:\n{str(e)}")
