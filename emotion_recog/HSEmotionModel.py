from hsemotion.facial_emotions import HSEmotionRecognizer
import numpy as np
from emotion_recog import EmotionModel
from scipy.special import softmax
from contextlib import redirect_stdout
from os import devnull


class HSEmotionModel(EmotionModel):
    emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    model_name = 'enet_b0_8_best_afew'

    def __init__(self, device='cpu', detector='retinaface'):
        super().__init__(detector=detector)
        with open(devnull, 'w') as f:
            with redirect_stdout(f):
                self.model = HSEmotionRecognizer(model_name=self.model_name, device=device)

    def __str__(self):
        return 'HSEmotionModel'

    def predict_vector(self, image, return_bbox=False):
        bbox = self._find_faces(image)
        faces = []
        for box in bbox:
            _, score_vector = self.model.predict_emotions(image[box['ymin']:box['ymax'],box['xmin']:box['xmax']], logits=True)
            faces.append(softmax(score_vector))
        if return_bbox:
            return faces, bbox
        return np.array(faces)

    def predict(self, image, return_bbox=False):
        faces, bbox = self.predict_vector(image, return_bbox=True)
        emotion = []
        for i, face in enumerate(faces):
            emotion.append(self.model.idx_to_class[np.argmax(face)])
        if return_bbox:
            return emotion, bbox
        return emotion

    def imshow(self, image):
        bbox = self._find_faces(image)
        preds = self.predict(image)
        return self.render_image(image, bbox, preds)
