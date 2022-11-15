import numpy as np
from rmn import RMN
from emotion_recog import EmotionModel
from collections import ChainMap


class RMNModel(EmotionModel):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self):
        super().__init__(detector=None)
        self.model = RMN()

    def __str__(self):
        return 'RMNModel'

    def predict_vector(self, image):
        results = self.model.detect_emotion_for_single_frame(image)
        preds = []
        for face in results:
            probas = dict(ChainMap(*face['proba_list']))
            preds.append(list(probas.items()))
        return np.array(preds)

    def predict(self, image):
        results = self.model.detect_emotion_for_single_frame(image)
        return np.array([face['emo_label'] for face in results])

    def imshow(self, image):
        results = self.model.detect_emotion_for_single_frame(image)
        preds = [r['emo_label'] for r in results]
        return self.render_image(image, results, preds)