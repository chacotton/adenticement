import os
import yaml
from emotion_recog.DeepFaceModel import DeepFaceModel
from emotion_recog.HSEmotionModel import HSEmotionModel
from emotion_recog.RMNModel import RMNModel
import matplotlib.pyplot as plt
import cv2
import argparse
import time
from collections import namedtuple
import numpy as np
from bbox_iou_eval import match_bboxes
import pandas as pd


model_score = namedtuple("model_score", "emotion bbox runtime")


def get_annotations(file):
    with open(file, 'r') as f:
        obj = yaml.safe_load(f)
    return obj

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


def image_test(models, image):
    preds = []
    for i, model in enumerate(models):
        start = time.time()
        pred, bbox = model.predict(image, return_bbox=True)
        runtime = time.time() - start
        preds.append(model_score(pred, bbox, runtime))
    return preds


def image_tester(img_dir, save=False):
    models = [DeepFaceModel(), HSEmotionModel(), RMNModel()]
    accuracy = [[], [], []]
    latency = [[], [], []]
    for ims in os.listdir(img_dir + '/images'):
        image = cv2.imread(img_dir + '/images/' + ims)
        labels = get_annotations(img_dir + '/annotations/' + ims[:-4] + 'txt')
        label_boxes = [e['bbox'] for e in labels['people']]
        preds = image_test(models, image)
        for i, pred in enumerate(preds):
            out = match_bboxes(label_boxes, pred.bbox)
            for y, yhat in zip(out[0], out[1]):
                accuracy[i].append(emotion_compare(labels['people'][y]['emotion'], pred.emotion[yhat]))
            latency[i].append(pred.runtime)
    df = pd.DataFrame(columns=[str(m) for m in models], index=['Accuracy', 'Latency'])
    df.loc['Accuracy', :] = [np.mean(x) for x in accuracy]
    df.loc['Latency', :] = [np.mean(x) for x in latency]
    if save:
        df.to_csv('model_test.csv')
    else:
        print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='path to image')
    parser.add_argument('-s', action='store_true', help='save images')
    args = parser.parse_args()
    if args.image.endswith('.jpeg'):
        image_viewer(cv2.imread(args.image), args.s)
    else:
        image_tester(args.image, args.s)


