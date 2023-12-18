import cv2
import torch
from PyQt6.QtCore import QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from picamera2 import Picamera2
from centralized.model import Model

class VideoProcessor(QThread):
    FPS = 60
    frame_processed = pyqtSignal(QPixmap, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (320, 240)}))  # Adjust size here
        self.picam2.start()
        self._face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.running = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(num_classes=7)
        checkpoint = torch.load("centralized/trained/private_model_234_65.t7")
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()
        self.model.to(self.device)

    def run(self):
        while self.running:
            frame = self.picam2.capture_array()
            processed_frame, emoji_path = self._detect_bounding_box(frame)
            pixmap = self._create_pixmap(processed_frame)
            self.frame_processed.emit(pixmap, emoji_path)
            self.msleep(int(1000/self.FPS))  # Delay to control frame rate

    def stop(self):
        self.running = False
        self.wait()

    def _detect_bounding_box(self, vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        faces = self._face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        predicted_emotion_label = None
        emoji_path = None

        largest_area = 0
        best_face = None

        for (x, y, w, h) in faces:
            # Calculate the area of the detected face
            area = w * h

            # Check if this face has a larger area than the previously detected ones
            if area > largest_area:
                largest_area = area
                best_face = (x, y, w, h)

        if best_face is not None:
            x, y, w, h = best_face
            face_roi = gray_image[y:y + h, x:x + w]  # Extract face region

            # Preprocess the face image for your PyTorch model
            face_roi = cv2.resize(face_roi, (44, 44))  # Assuming this size matches your model's input size
            face_roi = face_roi.astype('float32') / 255.0  # Normalize between 0 and 1

            # Convert to PyTorch tensor and perform inference
            face_tensor = torch.from_numpy(face_roi).unsqueeze(0).unsqueeze(0).to(self.device)
            predicted_emotion_logits = self.model(face_tensor)
            predicted_emotion = torch.argmax(predicted_emotion_logits, dim=1).item()

            emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            predicted_emotion_label = emotions[predicted_emotion]

            # Draw bounding box and text for the detected emotion
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(vid, predicted_emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Determine the emoji path based on the last detected face's emotion
            emoji_folder = 'gui/emojis/'
            emoji_dict = {
                'Angry': 'Angry.png',
                'Disgust': 'Disgust.png',
                'Fear': 'Fear.png',
                'Happy': 'Happy.png',
                'Sad': 'Sad.png',
                'Surprise': 'Surprise.png',
                'Neutral': 'Neutral.png'
            }
            emoji_path = emoji_folder + emoji_dict.get(predicted_emotion_label, '')  # Get emoji path

        return vid, emoji_path



    def _create_pixmap(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap