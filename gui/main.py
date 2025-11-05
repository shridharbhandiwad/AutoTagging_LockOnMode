"""Main GUI application."""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main():
    """Launch GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Airborne Track Behavior Tagger")
    app.setOrganizationName("Airborne Tracker Team")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
