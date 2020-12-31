import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from nltk import ngrams
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

warnings.filterwarnings('ignore')

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


def predict(y_train, X_test, lambda_spam, lambda_legit, prob):
    y0 = lambda_spam * priori_prob(y_train, SPAM) * likelihood_prob(X_test, SPAM, prob)
    y1 = lambda_legit * priori_prob(y_train, LEGIT) * likelihood_prob(X_test, LEGIT, prob)
    if y0 > y1:
        return 0
    return 1


def cross_validation(lambda_legit, alpha=1e-3, lambda_spam=1):
    X, y = load_dataset()
    CM = np.zeros((2, 2))

    for i in range(len(X)):
        ngrams, X_train = calc_ngrams(np.append(X[:i], X[i + 1:]), N)
        y_train = np.append((y[:i]), (y[i + 1:]))

        likelihood_y = calc_likelihood(X_train, y_train, alpha)

        y_test = [
            predict(y_train, calc_X_test(ngrams, xx, N), lambda_spam, lambda_legit, likelihood_y)
            for xx in X[i]
        ]

        CM += confusion_matrix(y_true=y[i], y_pred=y_test)
    return CM


def load_dataset():
    X = []
    y = []
    dirs = os.listdir("./data/")
    for dir in dirs:
        xx, yy = read_data("./data/" + dir + "/")
        X.append(xx)
        y.append(yy)

    X = np.array(X)
    y = np.array(y)
    return X, y


def solve():
    lambdas = [i for i in range(1, int(1e308), int(1e308 / 10))]
    ox = []
    oy = []
    acc = []

    for _lambda in tqdm(lambdas):
        # CM[true][pred]
        CM = cross_validation(_lambda)

        TN = CM[SPAM][SPAM]
        FN = CM[LEGIT][SPAM]
        TP = CM[LEGIT][LEGIT]
        FP = CM[SPAM][LEGIT]

        acc.append((TN + TP) / CM.sum())

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ox.append(TPR)
        oy.append(FPR)

    plt.plot(ox, oy)
    plt.title("roc")
    plt.show()

    plt.plot(lambdas, acc)
    plt.title('Accuracy')
    plt.show()


solve()
