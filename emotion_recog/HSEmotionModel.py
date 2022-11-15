from hsemotion.facial_emotions import HSEmotionRecognizer
import numpy as np
from emotion_recog import EmotionModel
from scipy.special import softmax


class HSEmotionModel(EmotionModel):
    emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    model_name = 'enet_b0_8_best_afew'

    def __init__(self, device='cpu', detector='retinaface'):
        super().__init__(detector=detector)
        self.model = HSEmotionRecognizer(model_name=self.model_name, device=device)

    def __str__(self):
        return 'HSEmotionModel'

    def predict_vector(self, image):
        bbox = self._find_faces(image)
        faces = []
        for box in bbox:
            _, score_vector = self.model.predict_emotions(image[box['ymin']:box['ymax'],box['xmin']:box['xmax']], logits=True)
            faces.append(softmax(score_vector))
        return np.array(faces)

    def predict(self, image):
        faces = self.predict_vector(image)
        emotion = []
        for i, face in enumerate(faces):
            emotion.append(self.model.idx_to_class[np.argmax(face)])
        return emotion

    def imshow(self, image):
        bbox = self._find_faces(image)
        preds = self.predict(image)
        return self.render_image(image, bbox, preds)
