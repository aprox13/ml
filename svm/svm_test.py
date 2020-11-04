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
from svm.svm2 import SVM2

FILE_MASK = "data/{0}.csv"
ITERATIONS = 3000

CHOOSE_BEST_THREADS = 12
C_CHOOSE = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def read_dataset(filename) -> DataSet:
    data = pd.read_csv(filename)
    X = data.values[:, :-1]
    tmp_y = data.values[:, -1]
    y = np.vectorize(lambda t: 1 if t == 'P' else -1)(tmp_y)
    return DataSet(X, y)


def pm_predict(f, data: np.ndarray, name):
    return pm(f, data, desc=f'{name}', max_workers=CHOOSE_BEST_THREADS, chunksize=25000)


def draw(clf: SVM, ds: DataSet, step):
    X = ds.get_X()
    y = ds.get_y()
    x_min, y_min = np.amin(X, 0)
    x_max, y_max = np.amax(X, 0)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    mesh_dots = np.c_[xx.ravel(), yy.ravel()]

    predict_z = np.array(pm_predict(clf.predict, mesh_dots, name='predict')).reshape(xx.shape)
    predict_soft_z = np.array(pm_predict(clf.predict_soft, mesh_dots, name='predict soft')).reshape(xx.shape)

    x0, y0 = X[y == -1].T
    x1, y1 = X[y == 1].T

    sup_ind = clf.get_support_indices()
    X_sup = X[sup_ind]
    x_sup, y_sup = X_sup.T

    bad = clf.get_bad_idx()
    if len(bad) != 0:
        X_bad = X[bad]

        x_bad, y_bad = X_bad.T

    def plot(_predict_z):
        plt.figure(figsize=(10, 10))
        plt.pcolormesh(xx, yy, _predict_z, cmap=plt.get_cmap('seismic'), shading='auto')
        plt.scatter(x0, y0, color='red', s=100)
        plt.scatter(x1, y1, color='blue', s=100)

        plt.scatter(x_sup, y_sup, color='white', marker='x', s=60)
        if len(bad) != 0:
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
    SVM2(max_iter=ITERATIONS, C=c, kernel=k)
    for k in KERNELS
    for c in C_CHOOSE
]


def accuracy(actual: np.ndarray, predicted: np.ndarray):
    return 1.0 - ((actual != predicted).sum() / float(actual.shape[0]))


def score(svm: SVM, ds: DataSet) -> float:
    cv = KFold(3)
    scores = []
    for train_index, test_index in cv.split(ds.get_X()):
        ds_ = ds.get_for_cross_validation(train_index, test_index)

        svm.fit(ds.get_X(), ds.get_y())

        y_pred = np.apply_along_axis(svm.predict, 1, ds_.get_test_X())
        scores.append(accuracy_score(ds_.get_test_y(), y_pred))
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
    ds = log_action("Reading", lambda: read_dataset(FILE_MASK.format("geyser")), with_start_msg=True)

    svm_best = SVMS[0]  # log_action("Choosing best svm", lambda: choose(ds, SVMS), with_start_msg=True)

    print(f"Got {svm_best}")
    log_action("trainig", lambda: svm_best.fit(ds.get_X(), ds.get_y()), with_start_msg=True)
    # svm_best.stat()
    log_action("drawing", lambda: draw(svm_best, ds, step=0.1))
