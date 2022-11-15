from emotion_recog.DeepFaceModel import DeepFaceModel
from emotion_recog.HSEmotionModel import HSEmotionModel
from emotion_recog.RMNModel import RMNModel
import matplotlib.pyplot as plt
import cv2
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='path to image')
    parser.add_argument('-s', action='store_true', help='save images')
    args = parser.parse_args()
    image = cv2.imread(args.image)
    models = [DeepFaceModel(), HSEmotionModel(), RMNModel()]
    for model in models:
        updated_image = model.imshow(image.copy())
        if not args.s:
            plt.imshow(updated_image)
            plt.show()
        else:
            cv2.imwrite(f'{str(model)}_{args.image.split("/")[-1]}', updated_image)
