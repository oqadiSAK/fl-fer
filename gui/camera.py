class Camera:
    CV_DEFAULT_CAMERA = 0
    PI_CAMERA_CONFIG = {"format": 'XRGB8888', "size": (320, 240)}

    def __init__(self, cam_type):
        if cam_type == "pi":
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            self._setup_pi_camera()
        elif cam_type == "cv":
            import cv2
            self.cap = cv2.VideoCapture(self.CV_DEFAULT_CAMERA)

    def _setup_pi_camera(self):
        self.picam2.configure(self.picam2.create_preview_configuration(main=self.PI_CAMERA_CONFIG))
        self.picam2.start()

    def capture_frame(self):
        if hasattr(self, 'picam2'):
            return self.picam2.capture_array()
        elif hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def release(self):
        if hasattr(self, 'cap'):
            self.cap.release()