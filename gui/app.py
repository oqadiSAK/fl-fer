import sys
import argparse
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

def app():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-type", choices=["pi", "cv"], required=True, 
                        help="Camera type to use: 'pi' for Picamera2, 'cv' for OpenCV VideoCapture")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(cam_type=args.cam_type)
    window.resize(480, 360)  
    window.show()
    sys.exit(app.exec())