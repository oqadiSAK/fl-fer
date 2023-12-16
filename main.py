import sys
import cv2
from picamera2 import Picamera2
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (320, 240)}))
        self.picam2.start()

        self._face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_detection_timer = QTimer(self)
        self.face_detection_timer.timeout.connect(self._perform_face_detection)
        self.face_detection_delay = 30  # Delay to perform face detection (in milliseconds)
        self.face_detection_timer.start(self.face_detection_delay)

        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Face Detection App")
        self._init_layouts()

    def _init_layouts(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.addLayout(self._init_video_layout(), 7)
        self.central_layout.addLayout(self._init_actions_layout(), 3)

    def _init_video_layout(self):
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        video_layout.addWidget(self.video_label)
        return video_layout

    def _init_actions_layout(self):
        actions_layout = QVBoxLayout()
        self.happy_label = QLabel("Happy")
        self.save_button = self._init_save_button()
        self.close_button_layout = self._init_close_button_layout()

        actions_layout.addWidget(self.happy_label, alignment=Qt.AlignmentFlag.AlignCenter)
        actions_layout.addWidget(self.save_button, alignment=Qt.AlignmentFlag.AlignCenter)
        actions_layout.addLayout(self.close_button_layout)
        return actions_layout

    def _init_save_button(self):
        save_button = QPushButton("SAVE")
        save_button.setFont(QFont("Arial", 14))
        save_button.setFixedSize(250, 250)
        return save_button

    def _init_close_button_layout(self):
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_button_layout = QHBoxLayout()
        close_button_layout.addStretch()
        close_button_layout.addWidget(close_button)
        return close_button_layout

    def _perform_face_detection(self):
        frame = self.picam2.capture_array()
        self._detect_bounding_box(frame)
        self._display_frame(frame)

    def _detect_bounding_box(self, vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        faces = self._face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        return faces

    def _display_frame(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label_size = self.video_label.size()
        pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.picam2.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
