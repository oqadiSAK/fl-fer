import cv2
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from centralized.model import Model

class VideoProcessor(QThread):
    FPS = 60
    FRAME_SIZE = (44, 44)
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    EMOJI_DICT = {emotion: f'{emotion}.png' for emotion in EMOTIONS}
    frame_processed = pyqtSignal(QPixmap, str)

    def __init__(self, cam_type, parent=None):
        super().__init__(parent)
        self.setup_camera(cam_type)
        self.setup_model()
        self.running = True

    def setup_camera(self, cam_type):
        if cam_type == "pi":
            self.setup_pi_camera()
        elif cam_type == "cv":
            self.setup_cv_camera()

    def setup_pi_camera(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (320, 240)}))  # Adjust size here
        self.picam2.start()

    def setup_cv_camera(self):
        self.cap = cv2.VideoCapture(0)  # 0 for default camera

    def setup_model(self):
        self._face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(num_classes=7)
        checkpoint = torch.load("centralized/trained/private_model_234_65.t7")
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()
        self.model.to(self.device)

    def run(self):
        while self.running:
            if hasattr(self, 'picam2'):
                frame = self.picam2.capture_array()
            elif hasattr(self, 'cap'):
                ret, frame = self.cap.read()
                if not ret:
                    continue
            processed_frame, emoji_path = self._detect_bounding_box(frame)
            pixmap = self._create_pixmap(processed_frame)
            self.frame_processed.emit(pixmap, emoji_path)
            self.msleep(int(1000/self.FPS))  # Delay to control frame rate
            
    def stop(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        self.wait()

    def _detect_bounding_box(self, vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        best_face = self._get_best_face(gray_image)
        vid, emoji_path = self._process_best_face(gray_image, vid, best_face)
        return vid, emoji_path

    def _get_best_face(self, gray_image):
        faces = self._face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        largest_area = 0
        best_face = None
        for (x, y, w, h) in faces:
            area = w * h
            if area > largest_area:
                largest_area = area
                best_face = (x, y, w, h)
        return best_face

    def _process_best_face(self, gray_image, vid, best_face):
        if best_face is not None:
            x, y, w, h = best_face
            face_roi = gray_image[y:y + h, x:x + w]
            predicted_emotion_label = self._predict_emotion(face_roi)
            vid, emoji_path = self._draw_bounding_box_and_get_emoji(vid, x, y, w, h, predicted_emotion_label)
            return vid, emoji_path
        return vid, None

    def _predict_emotion(self, face_roi):
        face_tensor = self._create_face_tensor(face_roi)
        predicted_emotion_logits = self.model(face_tensor)
        predicted_emotion = torch.argmax(predicted_emotion_logits, dim=1).item()
        predicted_emotion_label = self.EMOTIONS[predicted_emotion]
        return predicted_emotion_label

    def _create_face_tensor(self, face_roi):
        face_roi = cv2.resize(face_roi, self.FRAME_SIZE)
        face_roi = face_roi.astype('float32') / 255.0
        face_tensor = torch.from_numpy(face_roi).unsqueeze(0).unsqueeze(0).to(self.device)
        return face_tensor

    def _draw_bounding_box_and_get_emoji(self, vid, x, y, w, h, predicted_emotion_label):
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(vid, predicted_emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        emoji_folder = 'gui/emojis/'
        emoji_path = emoji_folder + self.EMOJI_DICT.get(predicted_emotion_label, '')
        return vid, emoji_path

    def _create_pixmap(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap