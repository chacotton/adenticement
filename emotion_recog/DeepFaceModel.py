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

    def predict_vector(self, image, return_bbox=False, localize=True):
        if localize:
            bbox = self._find_faces(image)
        else:
            bbox = [{'xmin': 0, 'ymin': 0, 'xmax': image.shape[1], 'ymax': image.shape[0]}]
        faces = []
        for box in bbox:
            img = image[self._image_box(box)]
            img = preprocess_face(img=img, **self.preproc_kwargs)
            preds = self.model.predict(img, verbose=0)[0, :]
            faces.append(softmax(preds))
        if return_bbox:
            return faces, bbox
        return np.array(faces)

    def predict(self, image, return_bbox=False, localize=True):
        faces, bbox = self.predict_vector(image, return_bbox=True, localize=localize)
        if return_bbox:
            return [self.emotions[np.argmax(f)] for f in faces], bbox
        return np.array([self.emotions[np.argmax(f)] for f in faces])


    def imshow(self, image):
        bbox = self._find_faces(image)
        preds = self.predict(image)
        image = self.render_image(image, bbox=bbox, preds=preds)
        return image






