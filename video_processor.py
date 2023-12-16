from PyQt6.QtCore import QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from picamera2 import Picamera2
import cv2

class VideoProcessor(QThread):
    FPS = 60
    frame_processed = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (320, 240)}))  # Adjust size here
        self.picam2.start()
        self._face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.running = True

    def run(self):
        while self.running:
            frame = self.picam2.capture_array()
            processed_frame = self._detect_bounding_box(frame)
            pixmap = self._create_pixmap(processed_frame)
            self.frame_processed.emit(pixmap)
            self.msleep(int(1000/self.FPS))  # Delay to control frame rate

    def stop(self):
        self.running = False
        self.wait()

    def _detect_bounding_box(self, vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        faces = self._face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        return vid

    def _create_pixmap(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap