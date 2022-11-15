from abc import ABC, abstractmethod
from deepface.detectors import FaceDetector
import cv2


class FaceDetectorModel:
    BACKENDS = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

    def __init__(self, detector):
        assert detector in self.BACKENDS, f"Pick a valid backend: {self.BACKENDS}"
        self.model = FaceDetector.build_model(detector)
        self.__name__ = detector

    def __str__(self):
        return self.__name__

    def detect_faces(self, img):
        faces = FaceDetector.detect_faces(self.model, str(self), img, align=False)
        regions = []
        for f in faces:
            regions.append({'xmin': int(f[1][0]), 'xmax': int(f[1][0] + f[1][2]),
                            'ymin': int(f[1][1]), 'ymax': int(f[1][1] + f[1][3])})
        return regions


class EmotionModel(ABC):
    emotions = []

    def __init__(self, detector=None):
        if detector is not None:
            self.detector = FaceDetectorModel(detector)
        else:
            self.detector = None

    @staticmethod
    def _image_box(bbox: dict, pad: int = 0, opencv: bool = True) -> tuple:
        x = slice(bbox['xmin'] - pad, bbox['xmax'] + pad)
        y = slice(bbox['ymin'] - pad, bbox['ymax'] + pad)
        if opencv:
            return y, x
        else:
            return x, y

    def render_image(self, image, bbox, preds):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 125)
        font_scale = .4
        image = cv2.putText(image, str(self), (0, 10), font, font_scale, color)
        for box, pred in zip(bbox, preds):
            image = cv2.rectangle(image, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), color=(0, 255, 0))
            image = cv2.putText(image, pred, (box['xmin'], box['ymin'] - 5), font, font_scale, color)
        return image


    def reset_face_detector(self, detector):
        self.detector = FaceDetectorModel(detector)

    def _find_faces(self, image):
        assert self.detector is not None, "Set a face detector before use"
        return self.detector.detect_faces(image)

    @abstractmethod
    def predict(self, image):
        raise NotImplementedError

    @abstractmethod
    def predict_vector(self, image):
        raise NotImplementedError
