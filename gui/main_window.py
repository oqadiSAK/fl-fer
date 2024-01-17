from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from gui.video_processor import VideoProcessor

class MainWindow(QMainWindow):
    SAVE_BUTTON_SIZE = 100
    SAVE_BUTTON_FONT = QFont("Arial", 14)
    EMOJI_LABEL_HEIGHT = 100
    VIDEO_LAYOUT_RATIO = 7
    ACTIONS_LAYOUT_RATIO = 3
    WINDOW_TITLE = "FL-FER"
    SAVE_BUTTON_TEXT = "SAVE"
    DEFAULT_EMOJI_LABEL_TEXT = "Emotion"

    def __init__(self, cam_type):
        super().__init__()
        self._init_ui(cam_type)
        self.video_processor.frame_processed.connect(self.update_frame)
        self.video_processor.start()

    def _init_ui(self, cam_type):
        self.setWindowTitle(self.WINDOW_TITLE)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_layout.addWidget(self.video_label)

        self.actions_layout = QVBoxLayout()
        self.emoji_label = QLabel() 
        self.save_button = self._create_save_button()
        self.actions_layout.addWidget(self.emoji_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.actions_layout.addWidget(self.save_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.addLayout(self.video_layout, self.VIDEO_LAYOUT_RATIO)
        self.central_layout.addLayout(self.actions_layout, self.ACTIONS_LAYOUT_RATIO)

        self.video_processor = VideoProcessor(cam_type=cam_type)

    def _create_save_button(self):
        save_button = QPushButton(self.SAVE_BUTTON_TEXT)
        save_button.setFont(self.SAVE_BUTTON_FONT)
        save_button.setFixedSize(self.SAVE_BUTTON_SIZE, self.SAVE_BUTTON_SIZE)
        return save_button

    def update_frame(self, pixmap, emoji_path):
        self.video_label.setPixmap(pixmap)
        self._update_emoji_label(emoji_path)

    def _update_emoji_label(self, emoji_path):
        if emoji_path:
            emoji_pixmap = QPixmap(emoji_path)
            self.emoji_label.setPixmap(emoji_pixmap.scaledToHeight(self.EMOJI_LABEL_HEIGHT))
        else:
            self.emoji_label.clear()
            self.emoji_label.setText(self.DEFAULT_EMOJI_LABEL_TEXT)

    def closeEvent(self, event):
        self.video_processor.stop()
        event.accept()