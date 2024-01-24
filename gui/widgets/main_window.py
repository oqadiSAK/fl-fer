from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from gui.driver_connection import DriverConnection
from gui.video_processor import VideoProcessor
from gui.widgets.video_layout import VideoLayout
from gui.widgets.actions_layout import ActionsLayout

class MainWindow(QMainWindow):
    WINDOW_TITLE = "FL-FER"
    VIDEO_LAYOUT_RATIO = 7
    ACTIONS_LAYOUT_RATIO = 3

    def __init__(self, model, device, cam_type, driver_ip, driver_port):
        super().__init__()
        self._init_ui(model, device, cam_type, driver_ip, driver_port)
        self.video_processor.frame_processed.connect(self.update_frame)
        self.video_processor.start()
        self.driver_connection.start()

    def _init_ui(self, model, device, cam_type, driver_ip, driver_port):
        self.video_processor = VideoProcessor(model, device, cam_type=cam_type)
        self.driver_connection = DriverConnection(driver_ip, driver_port)

        self.setWindowTitle(self.WINDOW_TITLE)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_layout = VideoLayout(self.video_processor)
        self.actions_layout = ActionsLayout(self.driver_connection, self.video_processor)

        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.addLayout(self.video_layout, self.VIDEO_LAYOUT_RATIO)
        self.central_layout.addLayout(self.actions_layout, self.ACTIONS_LAYOUT_RATIO)

    def update_frame(self, pixmap, emoji_path):
        self.video_layout.update_frame(pixmap)
        self.actions_layout.update_emoji_label(emoji_path)

    def closeEvent(self, event):
        self.video_processor.stop()
        event.accept()