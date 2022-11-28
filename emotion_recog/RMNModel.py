import numpy as np
from rmn import RMN
from emotion_recog import EmotionModel
from collections import ChainMap
from os import devnull
from contextlib import redirect_stdout


class RMNModel(EmotionModel):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self):
        super().__init__(detector=None)
        self.model = RMN()

    def __str__(self):
        return 'RMNModel'

    def detect(self, image):
        with open(devnull, 'w') as f:
            with redirect_stdout(f):
                pred = self.model.detect_emotion_for_single_frame(image)
        return pred

    def predict_vector(self, image, return_bbox=False, localize=True):
        results = self.detect(image)
        preds = []
        for face in results:
            probas = dict(ChainMap(*face['proba_list']))
            preds.append(list(probas.items()))
        if return_bbox:
            return preds, [{k: f[k] for k in f} for f in results]
        return np.array(preds)

    def predict(self, image, return_bbox=False, localize=True):
        results = self.detect(image)
        if return_bbox:
            return [f['emo_label'] for f in results], [{k: f[k] for k in f} for f in results]
        return np.array([face['emo_label'] for face in results])

    def imshow(self, image):
        results = self.detect(image)
        preds = [r['emo_label'] for r in results]
        return self.render_image(image, results, preds)
