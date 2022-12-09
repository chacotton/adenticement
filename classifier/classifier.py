import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train_model(n_splits):
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
    labeled_vecs = np.array(labeled_vecs)
    labeled_targets = np.array(labeled_targets)
    unlabeled_vecs = np.array(unlabeled_vecs)
    unlabeled_targets = np.array(unlabeled_targets)
    gau_avg_acc = []  # gaussian average accuracy
    knn_avg_acc = []  # knn average accuracy
    mlp_avg_acc = []  # mlp average accuracy
    svm_avg_acc = []  # svm average accuracy
    tree_avg_acc = []  # rbg average accuracy
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(labeled_vecs):
        x_train = labeled_vecs[train_index]
        x_test = labeled_vecs[test_index]
        y_train = labeled_targets[train_index]
        y_test = labeled_targets[test_index]
        gau_classifier = SelfTrainingClassifier(GaussianProcessClassifier()).fit(
            np.concatenate((x_train, unlabeled_vecs)),
            np.concatenate((y_train, unlabeled_targets)),
        )
        knn_classifier = SelfTrainingClassifier(KNeighborsClassifier()).fit(
            np.concatenate((x_train, unlabeled_vecs)),
            np.concatenate((y_train, unlabeled_targets)),
        )
        mlp_classifier = SelfTrainingClassifier(MLPClassifier()).fit(
            np.concatenate((x_train, unlabeled_vecs)),
            np.concatenate((y_train, unlabeled_targets)),
        )
        svm_classifier = SelfTrainingClassifier(SVC(probability=True)).fit(
            np.concatenate((x_train, unlabeled_vecs)),
            np.concatenate((y_train, unlabeled_targets)),
        )
        tree_classifier = SelfTrainingClassifier(DecisionTreeClassifier()).fit(
            np.concatenate((x_train, unlabeled_vecs)),
            np.concatenate((y_train, unlabeled_targets)),
        )
        gau_y_pred = gau_classifier.predict(x_test)
        gau_avg_acc.append(accuracy_score(y_test, gau_y_pred))
        knn_y_pred = knn_classifier.predict(x_test)
        knn_avg_acc.append(accuracy_score(y_test, knn_y_pred))
        mlp_y_pred = mlp_classifier.predict(x_test)
        mlp_avg_acc.append(accuracy_score(y_test, mlp_y_pred))
        svm_y_pred = svm_classifier.predict(x_test)
        svm_avg_acc.append(accuracy_score(y_test, svm_y_pred))
        tree_y_pred = tree_classifier.predict(x_test)
        tree_avg_acc.append(accuracy_score(y_test, tree_y_pred))
    return (
        gau_classifier,
        knn_classifier,
        mlp_classifier,
        svm_classifier,
        tree_classifier,
        gau_avg_acc,
        knn_avg_acc,
        mlp_avg_acc,
        svm_avg_acc,
        tree_avg_acc,
    )


if __name__ == "__main__":
    n_splits = 10
    (
        _,
        _,
        _,
        _,
        _,
        gau_avg_acc,
        knn_avg_acc,
        mlp_avg_acc,
        svm_avg_acc,
        tree_avg_acc,
    ) = train_model(n_splits)
    print(
        f"after {n_splits} times of iteration, the average accuracy is gaussian process: {sum(gau_avg_acc)/len(gau_avg_acc)}, knn: {sum(knn_avg_acc)/len(gau_avg_acc)}, mlp: {sum(mlp_avg_acc)/len(mlp_avg_acc)}, svm: {sum(svm_avg_acc)/len(svm_avg_acc)}, tree: {sum(tree_avg_acc)/len(tree_avg_acc)}"
    )
    plt.plot(gau_avg_acc, label="gaussian process")
    plt.plot(knn_avg_acc, label="k-neighbors")
    plt.plot(mlp_avg_acc, label="mlp")
    plt.plot(svm_avg_acc, label="svm")
    plt.plot(tree_avg_acc, label="decision tree")
    plt.legend()
    plt.show()
