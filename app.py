import sys
import argparse
import threading
import torch
from PyQt6.QtWidgets import QApplication
from client import start_client
from gui.main_window import MainWindow
from utils.loaders import load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-type", choices=["pi", "cv"], required=True, 
                        help="Camera type to use: 'pi' for Picamera2, 'cv' for OpenCV VideoCapture")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    flower_client_thread = threading.Thread(target=start_client, args=(model, device,))
    flower_client_thread.start()
    
    app = QApplication(sys.argv)
    window = MainWindow(model, device, cam_type=args.cam_type)
    window.resize(480, 360)  
    window.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()
