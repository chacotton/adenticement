import os
import cv2
import numpy as np
from engagement_model import EngagementModel
from tqdm import tqdm
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def get_classifier(file):
    if file is not None and os.path.exists(file):
        return file
    else:
        return 'classifier/clf.pkl'


def grapher(emotions, frame_rate, title):
    n = min(len(emotions), 60)
    time_elapsed = np.ceil(len(emotions) / frame_rate).astype(int)
    x = pd.date_range(start=datetime(2022,10,1,0,0,0), end=datetime(2022,10,1,0,time_elapsed,0), periods=frame_rate * time_elapsed)
    for i in range(emotions.shape[1]):
        plt.plot(x[:len(emotions)].strftime('%M:%S'), np.convolve(emotions[:, i], np.ones(n) / n, mode='same'))
    plt.legend()
    ax = plt.gca()
    ax.set_xticks(ax.get_xticks()[::10])
    plt.title(title)
    plt.savefig(f'{title.replace(" ", "-")}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Engagement Model Test')
    parser.add_argument('-c', '--classifier', type=str, help='path to classifier file, not required')
    parser.add_argument('-v', '--video', type=str, help='path to video file', required=True)
    parser.add_argument('-s', action='store_true', help='save graph output of Engagement Model')
    args = parser.parse_args()

    clf_file = get_classifier(args.classifier)
    model = EngagementModel(clf_file)
    if not os.path.exists(args.video):
        raise FileNotFoundError("Video File Does Not Exist!")
    cap = cv2.VideoCapture(args.video)
    engagement = []
    for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break
        else:
            engagement.append(model.predict(frame))
    engagement = np.array(engagement)
    np.save(Path(args.video).stem + '_engage.npy', engagement)
    if args.s:
        grapher(engagement, cap.get(cv2.CAP_PROP_FPS), title=f'{Path(args.video).stem} Reactions')
