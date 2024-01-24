import cv2
import os
import time
import pandas as pd

class FrameProcessor:
    FACE_DETECTION_SCALE_FACTOR = 1.1
    FACE_DETECTION_MIN_NEIGHBORS = 5
    SAVED_FRAMES_DIR = 'gui/local/saved_frames'
    LOCAL_SAVES_CSV = 'gui/local/local_saves.csv'
    
    def __init__(self, model):
        self.model = model

    def process_frame(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_face = self._get_best_face(gray_image)
        frame, predicted_emotion_label = self._process_best_face(gray_image, frame, best_face)
        return frame, predicted_emotion_label

    def save_frame(self, frame, emotion):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, self.model.FRAME_SIZE)
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
            
    def _get_best_face(self, gray_image):
        faces = self.model._face_classifier.detectMultiScale(gray_image, self.FACE_DETECTION_SCALE_FACTOR, 
                                                       self.FACE_DETECTION_MIN_NEIGHBORS, 
                                                       minSize=self.model.FRAME_SIZE)
        largest_area = 0
        best_face = None
        for (x, y, w, h) in faces:
            area = w * h
            if area > largest_area:
                largest_area = area
                best_face = (x, y, w, h)
        return best_face

    def _process_best_face(self, gray_image, frame, best_face):
        if best_face is not None:
            x, y, w, h = best_face
            face_roi = gray_image[y:y + h, x:x + w]
            predicted_emotion_label = self.model.predict_emotion(face_roi)
            frame = self._draw_bounding_box_and_get_emoji(frame, x, y, w, h, predicted_emotion_label)
            return frame, predicted_emotion_label
        return frame, None

    def _draw_bounding_box_and_get_emoji(self, frame, x, y, w, h, predicted_emotion_label):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(frame, predicted_emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame