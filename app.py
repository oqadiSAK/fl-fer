import sys
import argparse
import threading
from PyQt6.QtWidgets import QApplication
from app_fl_client import start_client
from gui.main_window import MainWindow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-type", choices=["pi", "cv"], required=True, 
                        help="Camera type to use: 'pi' for Picamera2, 'cv' for OpenCV VideoCapture")
    args = parser.parse_args()

    flower_client_thread = threading.Thread(target=start_client)
    flower_client_thread.start()
    
    app = QApplication(sys.argv)
    window = MainWindow(cam_type=args.cam_type)
    window.resize(480, 360)  
    window.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()
