from PyQt6.QtWidgets import QVBoxLayout, QLabel

class VideoLayout(QVBoxLayout):
    def __init__(self, video_processor):
        super().__init__()
        self.video_processor = video_processor
        self.video_label = QLabel()
        self.addWidget(self.video_label)

    def update_frame(self, pixmap):
        self.video_label.setPixmap(pixmap)