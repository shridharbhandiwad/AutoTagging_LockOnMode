"""File browser widget."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal


class FileBrowserWidget(QWidget):
    """Widget for browsing files."""
    
    file_selected = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout(self)
        
        self.label = QLabel("Drag and drop files here or click Browse")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        
        self.browse_btn = QPushButton("Browse...")
        layout.addWidget(self.browse_btn)
