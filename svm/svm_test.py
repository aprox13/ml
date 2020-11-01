from functools import partial
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import time
from tqdm.contrib.concurrent import process_map as pm
from functools import reduce

from utils.methods import *
from utils.plots import hist
from svm.smv import *

FILE_MASK = "data/{0}.csv"
ITERATIONS = 3000

CHOOSE_BEST_THREADS = 7
C_CHOOSE = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def in_range(x, start_ex, end_ex):
    return start_ex < x < end_ex


def pretty_time(millis: int) -> str:
    base = [(1000 * 60, "min"), (1000, "sec"), (1, "ms")]

    def step(acc, x):
        cur_millis, result = acc
        multiplier, name = x

        part = cur_millis // multiplier
        if part != 0:
            result.append(f"{part}{name}")
            cur_millis -= part * multiplier
            return cur_millis, result
        return acc

    res = reduce(step, base, (millis, []))[1]
    if len(res) != 0:
        return "".join(res)
    return "0ms"


def log_action(action_name, action, with_start_msg=False, with_result=True):
    def millis():
        return int(round(time.time() * 1000))

    if with_start_msg:
        print(f"starting '{action_name}'")

    start = millis()
    res = action()
    end_time_s = pretty_time(millis() - start)
    result_part = ""
    if with_result:
        result_part = f" with result {res}"

    print(f"'{action_name}' ends in {end_time_s}{result_part}")
    return res


class DataSet:
    def __init__(self, X: np.ndarray, y: np.ndarray, test_X=None, test_y=None):
        self.X = X
        self.y = y

        self._test_X = test_X
        self._test_y = test_y

    def shuffle(self):
        indices = np.arange(self.y.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_test_X(self):
        return self._test_X

    def get_test_y(self):
        return self._test_y

    def count(self):
        return self.y.shape[0]

    def get_for_cross_validation(self, train_indices, test_indices):
        train_X, test_X = self.X[train_indices], self.X[test_indices]
        train_y, test_y = self.y[train_indices], self.y[test_indices]

        return DataSet(X=train_X, y=train_y, test_X=test_X, test_y=test_y)


def read_dataset(filename) -> DataSet:
    data = pd.read_csv(filename)
    X = data.values[:, :-1]
    tmp_y = data.values[:, -1]
    y = np.vectorize(lambda t: 1 if t == 'P' else -1)(tmp_y)
    return DataSet(X, y)


def draw(clf, X, y, step):
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
    zz = np.apply_along_axis(lambda t: clf.predict(t), 1, mesh_dots)
    zz = np.array(zz).reshape(xx.shape)

    plt.figure(figsize=(10, 10))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    x0, y0 = X[y == -1].T
    x1, y1 = X[y == 1].T

    plt.pcolormesh(xx, yy, zz, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), shading='auto')
    plt.scatter(x0, y0, color='red', s=100)
    plt.scatter(x1, y1, color='blue', s=100)

    sup_ind = clf.get_suport_indices()
    X_sup = X[sup_ind]
    x_sup, y_sup = X_sup.T

    plt.scatter(x_sup, y_sup, color='white', marker='x', s=60)
    plt.show()


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


def unhot(function):
    def wrapper(actual, predicted):
        if len(actual.shape) > 1 and actual.shape[1] > 1:
            actual = actual.argmax(axis=1)
        if len(predicted.shape) > 1 and predicted.shape[1] > 1:
            predicted = predicted.argmax(axis=1)
        return function(actual, predicted)

    return wrapper


@unhot
def classification_error(actual: np.ndarray, predicted: np.ndarray):
    return (actual != predicted).sum() / float(actual.shape[0])


@unhot
def accuracy(actual, predicted):
    return 1.0 - classification_error(actual, predicted)


def score(svm: SVM, ds: DataSet) -> float:
    cv = KFold(4)
    scores = []
    for train_index, test_index in cv.split(ds.get_X()):
        cv_data_set = ds.get_for_cross_validation(train_index, test_index)

        svm.fit(ds.get_X(), ds.get_y())

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
    print("reading")
    ds = read_dataset(FILE_MASK.format("chips"))

    print("choose")
    a = choose(ds, SVMS)

    print(f"Got {a}")
