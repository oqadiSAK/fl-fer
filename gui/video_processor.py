import numpy as np
from PyQt6.QtCore import QThread, QMutex, QWaitCondition, pyqtSignal
from PyQt6.QtGui import QPixmap
from gui.camera import Camera
from gui.model import Model
from gui.frame_processor import FrameProcessor
from gui.utils import create_pixmap

class VideoProcessor(QThread):
    FPS = 30
    frame_processed = pyqtSignal(QPixmap, str)

    def __init__(self, model, device, cam_type, parent=None):
        super().__init__(parent)
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.paused = False
        self.camera = Camera(cam_type)
        self.model = Model(model, device)
        self.frame_processor = FrameProcessor(self.model)
        self.running = True
        self.frame = None
        self.last_emotion = None

    def run(self):
        while self.running:
            self.mutex.lock()
            if self.paused:
                self.wait_condition.wait(self.mutex)
            self.mutex.unlock()

            self.frame = self.camera.capture_frame()
            if self.frame is not None:
                processed_frame, emoji_path = self.frame_processor.process_frame(np.copy(self.frame))
                pixmap = create_pixmap(processed_frame)
                self.frame_processed.emit(pixmap, emoji_path)
            self.msleep(int(1000/self.FPS))  

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self.wait_condition.wakeAll()

    def stop(self):
        self.running = False
        self.camera.release()
        self.wait()
        
    def save_frame(self, emotion):
        self.frame_processor.save_frame(self.frame, emotion)
        self.last_emotion = emotion
        