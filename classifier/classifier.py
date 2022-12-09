import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

MODELS = {
    'svm': SVC(probability=True),
    'gaussian': GaussianProcessClassifier(),
    'knn': KNeighborsClassifier(),
    'mlp': MLPClassifier(),
    'dt': DecisionTreeClassifier(),
}

# initialize data
def get_data():
    labeled_vecs = []
    labeled_targets = []
    unlabeled_vecs = []
    unlabeled_targets = []
    with open(Path(__file__).parent.resolve() / Path("labels.json")) as f:
        part_targets = json.load(f)
    data = np.load(Path(__file__).parent.parent.resolve() / Path("emotion_vectors.npy"))
    row, _, col = data.shape
    data = data.reshape((row, col))
    labels_info = np.load(Path(__file__).parent.parent.resolve() / Path("emotion_vector_info.npy"))
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
    return labeled_vecs, labeled_targets, unlabeled_vecs, unlabeled_targets


def test_models(model_types=None, save_best=False, n_splits=10):
    if model_types is None:
        model_types = ['gaussian', 'knn', 'mlp', 'svm', 'dt']
    models = [MODELS[m] for m in model_types]
    accuracy = [0 for _ in models]
    labeled_vecs, labeled_targets, unlabeled_vecs, unlabeled_targets = get_data()
    epoch = 0
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(labeled_vecs):
        x_train = labeled_vecs[train_index]
        x_test = labeled_vecs[test_index]
        y_train = labeled_targets[train_index]
        y_test = labeled_targets[test_index]
        classifiers = [SelfTrainingClassifier(m).fit(
            np.concatenate((x_train, unlabeled_vecs)),
            np.concatenate((y_train, unlabeled_targets)),
        ) for m in models]
        preds = [c.predict(x_test) for c in classifiers]
        accuracy = [a + accuracy_score(y_test, p) for a, p in zip(accuracy, preds)]
        epoch += 1
    accuracy = np.array(accuracy) / epoch
    print('Accuracy')
    for m, acc in zip(model_types, accuracy):
        print(f'{m}: {acc * 100:.1f}%')
    if save_best:
        idx = np.argmax(accuracy)
        classifier = SelfTrainingClassifier(models[idx]).fit(
            np.concatenate((labeled_vecs, unlabeled_vecs)),
            np.concatenate((labeled_targets, unlabeled_targets)),
        )
        with open('clf.pkl', 'wb') as f:
            pickle.dump(classifier, f)
    return accuracy


if __name__ == '__main__':
    accuracy = test_models(save_best=True, n_splits=10)
    plt.plot(accuracy[0], label="gaussian process")
    plt.plot(accuracy[1], label="k-neighbors")
    plt.plot(accuracy[2], label="mlp")
    plt.plot(accuracy[3], label="svm")
    plt.plot(accuracy[4], label="decision tree")
    plt.legend()
    plt.show()
