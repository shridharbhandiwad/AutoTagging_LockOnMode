"""Simulator control widget."""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
                               QGroupBox, QFileDialog, QMessageBox, QCheckBox)
from PySide6.QtCore import Signal, QThread

from simulator.simulator import TrackSimulator, SimulationScenario


class SimulatorThread(QThread):
    """Thread for running simulation."""
    
    finished_signal = Signal(list)  # output files
    error = Signal(str)
    
    def __init__(self, scenario: SimulationScenario, output_dir: str):
        super().__init__()
        self.scenario = scenario
        self.output_dir = output_dir
    
    def run(self):
        """Run simulation."""
        try:
            simulator = TrackSimulator(self.scenario)
            output_files = simulator.run_simulation(self.output_dir)
            self.finished_signal.emit(output_files)
        except Exception as e:
            self.error.emit(str(e))


class SimulatorControlWidget(QWidget):
    """Widget for controlling simulator."""
    
    simulation_complete = Signal(list)  # output files
    
    def __init__(self):
        super().__init__()
        self.simulator_thread = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout(self)
        
        # Scenario configuration
        config_group = QGroupBox("Simulation Configuration")
        config_layout = QVBoxLayout()
        
        # Number of tracks
        tracks_layout = QHBoxLayout()
        tracks_layout.addWidget(QLabel("Number of Tracks:"))
        self.num_tracks_spin = QSpinBox()
        self.num_tracks_spin.setRange(1, 100)
        self.num_tracks_spin.setValue(10)
        tracks_layout.addWidget(self.num_tracks_spin)
        tracks_layout.addStretch()
        config_layout.addLayout(tracks_layout)
        
        # Duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration (seconds):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 600.0)
        self.duration_spin.setValue(60.0)
        duration_layout.addWidget(self.duration_spin)
        duration_layout.addStretch()
        config_layout.addLayout(duration_layout)
        
        # Update rate
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Update Rate (Hz):"))
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(1.0, 100.0)
        self.rate_spin.setValue(10.0)
        rate_layout.addWidget(self.rate_spin)
        rate_layout.addStretch()
        config_layout.addLayout(rate_layout)
        
        # Output format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['both', 'binary', 'csv'])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        config_layout.addLayout(format_layout)
        
        # Real-time mode
        self.realtime_check = QCheckBox("Real-time Mode")
        config_layout.addWidget(self.realtime_check)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_simulation)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
    
    def run_simulation(self):
        """Start simulation."""
        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "./data/simulated"
        )
        
        if not output_dir:
            return
        
        # Create scenario
        scenario = SimulationScenario(
            num_tracks=self.num_tracks_spin.value(),
            duration_seconds=self.duration_spin.value(),
            update_rate_hz=self.rate_spin.value(),
            output_format=self.format_combo.currentText(),
            real_time_mode=self.realtime_check.isChecked(),
        )
        
        # Start simulation thread
        self.simulator_thread = SimulatorThread(scenario, output_dir)
        self.simulator_thread.finished_signal.connect(self.on_simulation_complete)
        self.simulator_thread.error.connect(self.on_simulation_error)
        self.simulator_thread.start()
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Running simulation...")
    
    def stop_simulation(self):
        """Stop simulation."""
        if self.simulator_thread and self.simulator_thread.isRunning():
            self.simulator_thread.terminate()
            self.simulator_thread.wait()
        
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped")
    
    def on_simulation_complete(self, output_files: list):
        """Handle simulation completion."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Complete - generated {len(output_files)} files")
        
        self.simulation_complete.emit(output_files)
    
    def on_simulation_error(self, error_msg: str):
        """Handle simulation error."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Error")
        
        QMessageBox.critical(self, "Simulation Error", f"Simulation failed:\n{error_msg}")
