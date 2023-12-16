from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from video_processor import VideoProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._init_ui()
        self.video_processor = VideoProcessor()
        self.video_processor.frame_processed.connect(self.update_frame)
        self.video_processor.start()

    def _init_ui(self):
        self.setWindowTitle("Face Detection App")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._init_video_layout()
        self._init_actions_layout()
        self._init_main_layout()

    def _init_video_layout(self):
        self.video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_layout.addWidget(self.video_label)

    def _init_actions_layout(self):
        self.actions_layout = QVBoxLayout()
        self.happy_label = self._init_happy_label()
        self.save_button = self._init_save_button()
        self.actions_layout.addWidget(self.happy_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.actions_layout.addWidget(self.save_button, alignment=Qt.AlignmentFlag.AlignCenter)

    def _init_save_button(self):
        save_button = QPushButton("SAVE")
        save_button.setFont(QFont("Arial", 14))
        save_button.setFixedSize(100, 100)
        return save_button

    def _init_happy_label(self):
        happy_label = QLabel("Happy")
        return happy_label
    
    def _init_main_layout(self):
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.addLayout(self.video_layout, 7)
        self.central_layout.addLayout(self.actions_layout, 3)

    def update_frame(self, pixmap):
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.video_processor.stop()
        event.accept()