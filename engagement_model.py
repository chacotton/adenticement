from emotion_recog.DeepFaceModel import DeepFaceModel
import cv2
import pickle
import numpy as np


class EngagementModel:
    def __init__(self, file):
        self.emotion_model = DeepFaceModel()
        self.flip = False
        with open(file, 'rb') as f:
            self.classifier = pickle.load(f)

    def _video_mode(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.flip:
            img = cv2.flip(img, 0)
        return img

    def predict(self, img, video_mode=False):
        if video_mode:
            img = self._video_mode(img)
        people = self.emotion_model.predict_vector(img)
        if len(people) == 0:
            self.flip = True
            img = self._video_mode(img)
            people = self.emotion_model.predict_vector(img)
        try:
            pred = self.classifier.predict_proba(people)[:,1]
        except ValueError:
            pred = np.array([0])
        return pred

