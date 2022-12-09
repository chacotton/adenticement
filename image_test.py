import os
import yaml
from emotion_recog.DeepFaceModel import DeepFaceModel
from emotion_recog.HSEmotionModel import HSEmotionModel
from emotion_recog.RMNModel import RMNModel
import matplotlib.pyplot as plt
import cv2
import argparse
import time
import numpy as np
import pandas as pd


def emotion_compare(a, b):
    if a.lower() == 'happiness':
        a = 'happy'
    elif b.lower() == 'happiness':
        b = 'happy'
    return a.lower() == b.lower()

def image_viewer(image, save=False):
    models = [DeepFaceModel(), HSEmotionModel(), RMNModel()]
    for model in models:
        updated_image = model.imshow(image.copy())
        if not save:
            plt.imshow(updated_image)
            plt.show()
        else:
            cv2.imwrite(f'{str(model)}_{args.image.split("/")[-1]}', updated_image)


def image_tester(img_dir, save=False, max_images=None):
    models = [DeepFaceModel(), HSEmotionModel(), RMNModel()]
    df = pd.DataFrame(columns=[str(m) for m in models], index=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Latency'])
    latency = [[], [], []]
    for dirs in os.listdir(img_dir):
        accuracy = [[], [], []]
        for ims in os.listdir(img_dir + '/' + dirs)[:max_images]:
            image = cv2.imread(img_dir + '/' + dirs + '/' + ims)
            for i, m in enumerate(models):
                start = time.time()
                pred = m.predict(image, localize=False)
                end = time.time()
                latency[i].append(end - start)
                if len(pred) < 1:
                    accuracy[i].append(False)
                else:
                    accuracy[i].append(emotion_compare(pred[0], dirs))
        df.loc[dirs.capitalize(), :] = [np.mean(x) for x in accuracy]
    df.loc['Latency', :] = [np.mean(x) for x in latency]
    if save:
        df.to_csv('model_test.csv')
    else:
        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='path to image')
    parser.add_argument('-s', action='store_true', help='save images')
    args = parser.parse_args()
    if args.image.endswith('.jpeg'):
        image_viewer(cv2.imread(args.image), args.s)
    else:
        image_tester(args.image, args.s)


