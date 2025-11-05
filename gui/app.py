"""
Main application entry point.
"""
import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """Run the application"""
    app = QApplication(sys.argv)
    app.setApplicationName("Airborne Track Tagger")
    app.setOrganizationName("Radar Systems")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
