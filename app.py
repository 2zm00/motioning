import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply a simple dark style and font size for a generic look
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
        }
        QLabel {
            color: #ffffff;
            font-size: 14px;
        }
        QPushButton {
            background-color: #3c3f41;
            color: white;
            border: 1px solid #555555;
            padding: 8px;
            border-radius: 4px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #505354;
        }
        QListWidget {
            background-color: #3c3f41;
            color: white;
            font-size: 12px;
        }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
