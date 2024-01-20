import socket
from PyQt6.QtCore import QThread, pyqtSignal
from threading import Event
from logging import INFO, ERROR
from flwr.common.logger import log

class DriverConnection(QThread):
    fl_started = pyqtSignal()
    fl_ended = pyqtSignal()
    ready = pyqtSignal()
    waiting = pyqtSignal()
    status_changed = pyqtSignal(str)
    
    def __init__(self, address='192.168.1.102', port=9093):
        super().__init__()
        self.address = address
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected_event = Event()

    def run(self):
        log(INFO, f"Connecting to driver at {self.address}:{self.port}")
        try:
            self.client_socket.connect((self.address, self.port))
            self.connected_event.set()
            log(INFO, f"Connected to driver")
        except Exception as e:
            log(ERROR, f"Failed to connect to driver: {e}")
            self.status_changed.emit("FL_ERROR")  # Emit signal with "FL_ERROR"
            return
        while True:
            try:
                data = self.client_socket.recv(1024)
                if data == b'READY':
                    log(INFO, f"Received READY from driver")
                    self.ready.emit()
                    self.status_changed.emit("READY")  # Emit signal with "READY"
                elif data == b'WAITING':
                    log(INFO, f"Received WAITING from driver")
                    self.waiting.emit() 
                    self.status_changed.emit("WAITING")  # Emit signal with "WAITING"
                elif data == b'FL_STARTED':
                    log(INFO, f"Received FL_STARTED from driver")
                    self.fl_started.emit()
                    self.status_changed.emit("FL_RUNNING")  # Emit signal with "FL_RUNNING"
                elif data == b'FL_ENDED':
                    log(INFO, f"Received FL_ENDED from driver")
                    self.fl_ended.emit()
                    self.status_changed.emit("IDLE")  # Emit signal with "IDLE"
            except ConnectionResetError:
                log(ERROR, "Driver has disconnected")
                self.status_changed.emit("FL_ERROR")  # Emit signal with "FL_ERROR"
                break
            except Exception as e:
                log(ERROR, f"Failed to read from driver: {e}")
                self.status_changed.emit("FL_ERROR")  # Emit signal with "FL_ERROR"
                break

    def trigger_fl(self):
        log(INFO, f"Sending TRIGGER_FL to driver")
        self.connected_event.wait()
        try:
            self.client_socket.sendall(b'TRIGGER_FL')
            log(INFO, f"Sent TRIGGER_FL to driver")
        except Exception as e:
            log(ERROR, f"Failed to send TRIGGER_FL to driver: {e}")