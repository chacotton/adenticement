import json

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

# initialize data
labeled_vecs = []
labeled_targets = []
unlabeled_vecs = []
unlabeled_targets = []
with open("labels.json") as f:
    part_targets = json.load(f)
data = np.load("../emotion_vectors.npy")
row, _, col = data.shape
data = data.reshape((row, col))
labels_info = np.load("../emotion_vector_info.npy")
for vec, (filename, res) in zip(data, labels_info):
    if filename in part_targets:
        labeled_vecs.append(vec)
        labeled_targets.append(part_targets.get(filename))
    else:
        unlabeled_vecs.append(vec)
        unlabeled_targets.append(-1)

avg_acc = 0  # average accuracy
epoch = 10  # iteration times
for i in range(epoch):
    x_train, x_test, y_train, y_test = train_test_split(
        labeled_vecs, labeled_targets)
    classifier = SelfTrainingClassifier(
        SVC(probability=True)).fit(x_train+unlabeled_vecs, y_train+unlabeled_targets)
    y_pred = classifier.predict(x_test)
    avg_acc += accuracy_score(y_test, y_pred)
avg_acc /= epoch
print(f"after {epoch} times of iteration, the average accuracy is {avg_acc}")
