from PyQt6.QtWidgets import QVBoxLayout, QLabel, QPushButton, QDialog, QRadioButton
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtCore import Qt

class ActionsLayout(QVBoxLayout):
    SAVE_BUTTON_SIZE = 100
    SAVE_BUTTON_FONT = QFont("Arial", 14)
    TRIGGER_FL_BUTTON_SIZE_X = 100
    TRIGGER_FL_BUTTON_SIZE_Y = 50
    TRIGGER_FL_BUTTON_FONT = QFont("Arial", 7)
    EMOJI_LABEL_HEIGHT = 100
    SAVE_BUTTON_TEXT = "SAVE"
    DEFAULT_EMOJI_LABEL_TEXT = "Emotion"

    def __init__(self, driver_connection, video_processor):
        super().__init__()
        self.driver_connection = driver_connection
        self.video_processor = video_processor
        self.emoji_label = QLabel()
        self.status_label = QLabel("IDLE")
        self.driver_connection.status_changed.connect(self.status_label.setText)
        self.trigger_fl_button = self._create_trigger_fl_button()
        self.driver_connection.fl_ended.connect(lambda: self.trigger_fl_button.setEnabled(True))
        self.driver_connection.fl_started.connect(lambda: self.trigger_fl_button.setDisabled(True))
        self.driver_connection.ready.connect(lambda: self.trigger_fl_button.setEnabled(True))
        self.driver_connection.waiting.connect(lambda: self.trigger_fl_button.setDisabled(True))
        self.save_button = self._create_save_button()
        self.addWidget(self.emoji_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.addWidget(self.trigger_fl_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.addWidget(self.save_button, alignment=Qt.AlignmentFlag.AlignCenter)

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

    def update_emoji_label(self, emoji_label):
        emoji_path = f"gui/emojis/{emoji_label}.png"
        if emoji_label:
            emoji_pixmap = QPixmap(emoji_path)
            self.emoji_label.setPixmap(emoji_pixmap.scaledToHeight(self.EMOJI_LABEL_HEIGHT))
        else:
            self.emoji_label.clear()
            self.emoji_label.setText(self.DEFAULT_EMOJI_LABEL_TEXT)