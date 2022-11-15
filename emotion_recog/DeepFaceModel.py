from deepface import DeepFace
from deepface.commons.functions import preprocess_face
from emotion_recog import EmotionModel
from scipy.special import softmax
import numpy as np


class DeepFaceModel(EmotionModel):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    preproc_kwargs = {
        'target_size': (48, 48),
        'grayscale': True,
        'enforce_detection': False,
        'detector_backend': 'opencv',
        'return_region': False
                      }

    def __init__(self, detector='retinaface'):
        super().__init__(detector=detector)
        self.model = DeepFace.build_model('Emotion')

    def __str__(self):
        return 'DeepFaceModel'

    def predict_vector(self, image):
        bbox = self._find_faces(image)
        faces = []
        for box in bbox:
            img = image[self._image_box(box)]
            img = preprocess_face(img=img, **self.preproc_kwargs)
            preds = self.model.predict(img, verbose=0)[0, :]
            faces.append(softmax(preds))
        return np.array(faces)

    def predict(self, image):
        faces = self.predict_vector(image)
        return np.array([self.emotions[np.argmax(f)] for f in faces])

    def imshow(self, image):
        bbox = self._find_faces(image)
        preds = self.predict(image)
        image = self.render_image(image, bbox=bbox, preds=preds)
        return image






