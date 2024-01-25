import sys
import argparse
import threading
import torch
from PyQt6.QtWidgets import QApplication
from fl.client import start_client
from gui.widgets.main_window import MainWindow
from utils.loaders import load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-type", choices=["pi", "cv"], required=True, 
                        help="Camera type to use: 'pi' for Picamera2, 'cv' for OpenCV VideoCapture")
    parser.add_argument('--server_ip', default='localhost', type=str, help='Server address')
    parser.add_argument('--server_port', default=9092, type=int, help='Server port')
    parser.add_argument('--driver_ip', default='localhost', type=str, help='Driver address')
    parser.add_argument('--driver_port', default=9093, type=int, help='Driver port')
    parser.add_argument('--threshold', default=5, type=int, 
                        help='Threshold value where represents the minimum number of images to participate in the federated learning')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    server_address = f"{args.server_ip}:{args.server_port}"
    flower_client_thread = threading.Thread(target=start_client, args=(model, device, server_address, args.threshold, ))
    flower_client_thread.start()
    
    app = QApplication(sys.argv)
    window = MainWindow(model, device, args.cam_type, args.driver_ip, args.driver_port, args.threshold)
    window.resize(480, 360)  
    window.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()
