from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QDialog, QRadioButton
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from gui.driver_connection import DriverConnection
from gui.video_processor import VideoProcessor

class MainWindow(QMainWindow):
    SAVE_BUTTON_SIZE = 100
    SAVE_BUTTON_FONT = QFont("Arial", 14)
    TRIGGER_FL_BUTTON_SIZE_X = 100
    TRIGGER_FL_BUTTON_SIZE_Y = 50
    TRIGGER_FL_BUTTON_FONT = QFont("Arial", 7)
    EMOJI_LABEL_HEIGHT = 100
    VIDEO_LAYOUT_RATIO = 7
    ACTIONS_LAYOUT_RATIO = 3
    WINDOW_TITLE = "FL-FER"
    SAVE_BUTTON_TEXT = "SAVE"
    DEFAULT_EMOJI_LABEL_TEXT = "Emotion"

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

        self.video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_layout.addWidget(self.video_label)

        self.actions_layout = QVBoxLayout()
        self.emoji_label = QLabel() 
        self.status_label = QLabel("IDLE")
        self.driver_connection.status_changed.connect(self.status_label.setText)
        self.trigger_fl_button = self._create_trigger_fl_button()
        self.driver_connection.fl_ended.connect(lambda: self.trigger_fl_button.setEnabled(True))
        self.driver_connection.fl_started.connect(lambda: self.trigger_fl_button.setDisabled(True))
        self.driver_connection.ready.connect(lambda: self.trigger_fl_button.setEnabled(True))
        self.driver_connection.waiting.connect(lambda: self.trigger_fl_button.setDisabled(True))
        self.save_button = self._create_save_button()
        self.actions_layout.addWidget(self.emoji_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.actions_layout.addWidget(self.trigger_fl_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.actions_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.actions_layout.addWidget(self.save_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.addLayout(self.video_layout, self.VIDEO_LAYOUT_RATIO)
        self.central_layout.addLayout(self.actions_layout, self.ACTIONS_LAYOUT_RATIO)
    
    def _create_trigger_fl_button(self):
        trigger_fl_button = QPushButton("TRIGGER FL")
        trigger_fl_button.setFont(self.TRIGGER_FL_BUTTON_FONT)
        trigger_fl_button.setFixedSize(self.TRIGGER_FL_BUTTON_SIZE_X, self.TRIGGER_FL_BUTTON_SIZE_Y)
        trigger_fl_button.clicked.connect(self.driver_connection.trigger_fl)
        return trigger_fl_button
    
    def _create_save_button(self):
        save_button = QPushButton(self.SAVE_BUTTON_TEXT)
        save_button.setFont(self.SAVE_BUTTON_FONT)
        save_button.setFixedSize(self.SAVE_BUTTON_SIZE, self.SAVE_BUTTON_SIZE)
        save_button.clicked.connect(self.save_frame_and_select_emotion)
        return save_button

    def save_frame_and_select_emotion(self):
        self.show_emotion_dialog()
        
    def show_emotion_dialog(self):
        self.video_processor.pause()
        dialog = QDialog()
        dialog.setWindowTitle("Select Emotion")
        dialog.resize(400, 300)
        layout = QVBoxLayout()
        dialog.setLayout(layout)

        radio_buttons = []
        for emotion in self.video_processor.EMOTIONS:
            radio_button = QRadioButton(emotion)
            layout.addWidget(radio_button)
            radio_buttons.append(radio_button)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(lambda: self.save_frame(dialog, radio_buttons))
        layout.addWidget(ok_button)

        for radio_button in radio_buttons:
            if radio_button.text() == self.video_processor.last_emotion:
                radio_button.setChecked(True)
                break

        dialog.exec()
        self.video_processor.resume()
    
    def save_frame(self, dialog, radio_buttons):
        selected_emotion_index = next((i for i, rb in enumerate(radio_buttons) if rb.isChecked()))
        self.video_processor.save_frame(selected_emotion_index)
        dialog.accept()
            
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