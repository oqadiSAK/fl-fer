import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(480, 360)  
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
