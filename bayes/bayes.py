import os
import re
from functools import partial
from multiprocessing.spawn import freeze_support

import matplotlib.pyplot as plt
import numpy as np
from nltk import ngrams
from sklearn.metrics import confusion_matrix
from tqdm import trange
from tqdm.contrib.concurrent import process_map as pm

LEGIT: int = 1
SPAM: int = 0

N = 1


def read_data(folder):
    X = []
    y = np.array([])
    files = os.listdir(folder)
    for file in files:
        with open(os.path.join(folder, file)) as f:
            subject = f.readline().rstrip().split(' ')
            subject = list(map(int, subject[1:]))
            f.readline()
            data = list(map(int, f.readline().split(' ')))
            X.append(subject + data)
            y = np.append(y, LEGIT if re.search(r'legit', file) else SPAM)
    X = np.array(X)
    return X, y


def calc_ngrams(X, n):
    set_ng = set()
    for x in X:
        set_ng.update(ngrams(x, n))
    ngram = []
    for x in X:
        ng = dict.fromkeys(set_ng, 0)
        for i in range(0, len(x) - n):
            ng[tuple(x[i:i + n])] += 1
        ngram.append(list(ng.values()))
    return set_ng, ngram


def calc_X_test(keys, X_test, n):
    ng = dict.fromkeys(keys, 0)
    for i in range(0, len(X_test) - n):
        key = tuple(X_test[i:i + n])
        if key in keys:
            ng[key] += 1

    return np.array(list(ng.values()))


def calc_prob(X, y, alpha, f):
    new_X = []
    res = []
    for i in range(len(y)):
        if y[i] == f:
            new_X.append(X[i])
    for i in range(len(X[0])):
        sum = 0
        for x in new_X:
            if x[i] > 0:
                sum += 1
        res.append((sum + alpha) / (alpha * 2 + len(new_X)))
    return res


def calc_likelihood(X, y, alpha):
    return np.array([calc_prob(X, y, alpha, 0), calc_prob(X, y, alpha, 1)])


def priori_prob(y_train, y):
    return np.sum(y_train == y) / len(y_train)


def likelihood_prob(X_test, y, prob):
    p = np.float64(1.0)
    count = 0
    for i in range(len(X_test)):
        if X_test[i] > 0:
            p *= prob[y][i]
            count += 1
        else:
            p *= (1 - prob[y][i])
    return p


def predict(y_train, X_test, _lambda, prob):
    y0 = _lambda[0] * priori_prob(y_train, 0) * likelihood_prob(X_test, 0, prob)
    y1 = _lambda[1] * priori_prob(y_train, 1) * likelihood_prob(X_test, 1, prob)
    if y0 > y1:
        return 0
    return 1


def cross_validation(l_max, X, y, alpha=1e-3):
    l = (1, l_max)
    acc = []
    true_pred, false_pred, false_legit = 0, 0, 0
    k0, k1 = 0, 0
    for i in trange(len(X), desc='Main loop'):
        ngrams, X_train = calc_ngrams(np.append(X[:i], X[i + 1:]), N)
        y_train = np.append((y[:i]), (y[i + 1:]))

        y_control = y[i]

        likelihood_y = calc_likelihood(X_train, y_train, alpha)

        y_test = []
        for j in range(len(X[i])):
            X_test = calc_X_test(ngrams, X[i][j], N)
            y_test.append(predict(y_train, X_test, l, likelihood_y))

        CM = confusion_matrix(y_true=y_control, y_pred=y_test)

        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]

        true_pred += TP
        false_pred += FP
        false_legit += FN

        acc.append((TN + TP) / np.sum(CM))

        k0 += np.sum(y_control == SPAM)
        k1 += np.sum(y_control == LEGIT)
    return np.mean(acc), false_legit, true_pred / k1, false_pred / k0


def solve():
    X = []
    y = []
    dirs = os.listdir("./data/")
    for dir in dirs:
        xx, yy = read_data("./data/" + dir + "/")
        X.append(xx)
        y.append(yy)

    X = np.array(X)
    y = np.array(y)

    lambdaa = [i for i in range(1, int(1e308), int(1e308 / 10))]
    ox = []
    oy = []

    res = map(partial(cross_validation, X=X, y=y), lambdaa)

    for (acc, false_legit, true_pred, false_pred) in res:
        ox.append(true_pred)
        oy.append(false_pred)

    plt.plot(ox, oy)
    plt.title("roc")
    plt.show()


solve()
