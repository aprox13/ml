from functools import partial
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from tqdm.contrib.concurrent import process_map as pm

from utils.data_set import DataSet
from utils.methods import *
from utils.plots import hist
from svm.smv import *

FILE_MASK = "data/{0}.csv"
ITERATIONS = 3000

CHOOSE_BEST_THREADS = 7
C_CHOOSE = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def read_dataset(filename) -> DataSet:
    data = pd.read_csv(filename)
    X = data.values[:, :-1]
    tmp_y = data.values[:, -1]
    y = np.vectorize(lambda t: 1 if t == 'P' else -1)(tmp_y)
    return DataSet(X, y)


def draw(clf: SVM, ds: DataSet, step):
    X = ds.get_X()
    y = ds.get_y()
    stepx = step
    stepy = 0.01
    x_min, y_min = np.amin(X, 0)
    x_max, y_max = np.amax(X, 0)
    x_min -= stepx
    x_max += stepx
    y_min -= stepy
    y_max += stepy
    xx, yy = np.meshgrid(np.arange(x_min, x_max, stepx),
                         np.arange(y_min, y_max, stepy))

    mesh_dots = np.c_[xx.ravel(), yy.ravel()]
    predict_z = np.array([clf.predict(v) for v in tqdm(mesh_dots, desc="predict")]).reshape(xx.shape)
    predict_soft_z = np.array([clf.predict_soft(v) for v in tqdm(mesh_dots, desc="Soft predict")]).reshape(xx.shape)

    plt.figure(figsize=(10, 10))
    x0, y0 = X[y == -1].T
    x1, y1 = X[y == 1].T

    sup_ind = clf.get_suport_indices()
    X_sup = X[sup_ind]
    x_sup, y_sup = X_sup.T

    bad = clf.get_bad_idx()
    X_bad = X[bad]

    x_bad, y_bad = X_bad.T

    def plot(_predict_z):
        plt.pcolormesh(xx, yy, _predict_z, cmap=plt.get_cmap('seismic'), shading='auto')
        plt.scatter(x0, y0, color='red', s=100)
        plt.scatter(x1, y1, color='blue', s=100)

        plt.scatter(x_sup, y_sup, color='white', marker='x', s=60)
        plt.scatter(x_bad, y_bad, color='black', marker='X', s=60)
        plt.show()

    plot(predict_z)
    plot(predict_soft_z)


KERNELS = [
    Linear(),
    RBF(1),
    RBF(2),
    RBF(3),
    RBF(4),
    RBF(5),
    Poly(2),
    Poly(3),
    Poly(4),
    Poly(5)
]

SVMS = [
    SVM(max_iter=ITERATIONS, C=c, kernel=k)
    for k in KERNELS
    for c in C_CHOOSE
]


def accuracy(actual: np.ndarray, predicted: np.ndarray):
    return 1.0 - ((actual != predicted).sum() / float(actual.shape[0]))


def score(svm: SVM, ds: DataSet) -> float:
    cv = KFold(4)
    scores = []
    for train_index, test_index in cv.split(ds.get_X()):
        cv_data_set = ds.get_for_cross_validation(train_index, test_index)

        svm.fit(cv_data_set.get_X(), cv_data_set.get_y())

        y_pred = np.apply_along_axis(svm.predict, 1, cv_data_set.get_test_X())
        scores.append(accuracy(cv_data_set.get_test_y(), y_pred))
    return np.average(np.array(scores))


def draw_best_hist(scores, svms):
    zipped = zip(scores, svms)
    mp = group_by(lambda x: str(x[1].kernel), zipped)

    def by_c(lst: List[Tuple[float, SVM]]):
        return [find_in(lst, lambda x: x[1].C == C, default=0)[0] for C in C_CHOOSE]

    res = {}
    for k in mp.keys():
        res[k] = by_c(mp[k])

    hist(
        res,
        index=C_CHOOSE,
        title='Accuracy by C',
        x_label='C',
        y_label='Accuracy'
    )


def choose(data_set: DataSet, svms: List[SVM]):
    scores = pm(
        partial(score, ds=data_set),
        svms,
        max_workers=CHOOSE_BEST_THREADS
    )

    draw_best_hist(scores, svms)

    # noinspection PyTypeChecker
    return svms[
        np.argmax(
            np.array(
                scores
            )
        )
    ]


if __name__ == '__main__':
    ds = log_action("Reading", lambda: read_dataset(FILE_MASK.format("chips")), with_start_msg=True)

    svm_best = SVM(max_iter=ITERATIONS, C=50,
                   kernel=RBF(1))  # log_action("Choosing best svm", lambda: choose(ds, SVMS), with_start_msg=True)

    print(f"Got {svm_best}")
    log_action("trainig", lambda: svm_best.fit(ds.get_X(), ds.get_y()), with_start_msg=True)
    svm_best.stat()
    log_action("drawing", lambda: draw(svm_best, ds, step=0.0005))
