import cv2
import torch
import time
import pandas as pd
import os
import numpy as np
from PyQt6.QtCore import QThread, QMutex, QWaitCondition, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from centralized.model import Model

class VideoProcessor(QThread):
    FPS = 60
    FRAME_SIZE = (44, 44)
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    EMOJI_DICT = {emotion: f'{emotion}.png' for emotion in EMOTIONS}
    frame_processed = pyqtSignal(QPixmap, str)
    CV_DEFAULT_CAMERA = 0
    PI_CAMERA_CONFIG = {"format": 'XRGB8888', "size": (320, 240)}
    HAARCASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    MODEL_CHECKPOINT_PATH = "centralized/trained/private_model_234_65.t7"
    SAVED_FRAMES_DIR = 'gui/local/saved_frames'
    LOCAL_SAVES_CSV = 'gui/local/local_saves.csv'
    EMOJI_FOLDER = 'gui/emojis/'
    FACE_DETECTION_SCALE_FACTOR = 1.1
    FACE_DETECTION_MIN_NEIGHBORS = 5
    FACE_DETECTION_MIN_SIZE = (40, 40)

    def __init__(self, cam_type, parent=None):
        super().__init__(parent)
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.paused = False
        self._setup_camera(cam_type)
        self._setup_model()
        self.running = True
        self.frame = None
        self.last_emotion = None
        
    def run(self):
        while self.running:
            self.mutex.lock()
            if self.paused:
                self.wait_condition.wait(self.mutex)
            self.mutex.unlock()
                
            if hasattr(self, 'picam2'):
                self.frame = self.picam2.capture_array()
            elif hasattr(self, 'cap'):
                ret, self.frame = self.cap.read()
                if not ret:
                    continue
            processed_frame, emoji_path = self._detect_bounding_box(np.copy(self.frame))
            pixmap = self._create_pixmap(processed_frame)
            self.frame_processed.emit(pixmap, emoji_path)
            self.msleep(int(1000/self.FPS))  # Delay to control frame rate
    
    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self.wait_condition.wakeAll()
           
    def stop(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        self.wait()

    def save_frame(self, emotion):
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, VideoProcessor.FRAME_SIZE)
        timestamp = time.strftime("%Y%m%d-%H%M%S")  

        if not os.path.isdir(self.SAVED_FRAMES_DIR):
            os.makedirs(self.SAVED_FRAMES_DIR)

        cv2.imwrite(f'{self.SAVED_FRAMES_DIR}/frame_{timestamp}.png', resized_frame) 
        pixels = ' '.join(map(str, resized_frame.flatten()))
        data = pd.DataFrame([[emotion, pixels]], columns=['emotion', 'pixels'])

        if not os.path.isfile(self.LOCAL_SAVES_CSV):
            data.to_csv(self.LOCAL_SAVES_CSV, mode='w', index=False)
        else:
            data.to_csv(self.LOCAL_SAVES_CSV, mode='a', index=False, header=False)
    
    def _setup_camera(self, cam_type):
        if cam_type == "pi":
            self._setup_pi_camera()
        elif cam_type == "cv":
            self._setup_cv_camera()

    def _setup_pi_camera(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main=self.PI_CAMERA_CONFIG))
        self.picam2.start()
        
    def _setup_cv_camera(self):
        self.cap = cv2.VideoCapture(self.CV_DEFAULT_CAMERA)

    def _setup_model(self):
        self._face_classifier = cv2.CascadeClassifier(self.HAARCASCADE_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(num_classes=7)
        checkpoint = torch.load("centralized/trained/private_model_234_65.t7")
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()
        self.model.to(self.device)
        
    def _detect_bounding_box(self, vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        best_face = self._get_best_face(gray_image)
        vid, emoji_path = self._process_best_face(gray_image, vid, best_face)
        return vid, emoji_path

    def _get_best_face(self, gray_image):
        faces = self._face_classifier.detectMultiScale(gray_image, self.FACE_DETECTION_SCALE_FACTOR, 
                                                       self.FACE_DETECTION_MIN_NEIGHBORS, 
                                                       minSize=self.FACE_DETECTION_MIN_SIZE)
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
            self.last_emotion = predicted_emotion_label
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
        emoji_path = self.EMOJI_FOLDER + self.EMOJI_DICT.get(predicted_emotion_label, '')
        return vid, emoji_path

    def _create_pixmap(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap