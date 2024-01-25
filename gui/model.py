import cv2
import torch

class Model:
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    HAARCASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    FRAME_SIZE = (48, 48)
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._face_classifier = cv2.CascadeClassifier(self.HAARCASCADE_PATH)
        self.model.eval()

    def predict_emotion(self, face_roi):
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