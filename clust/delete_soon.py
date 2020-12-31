import os
import re

import numpy as np
from nltk import ngrams
import matplotlib.pyplot as plt


def read_data(folder):
    X = []
    y = np.array([])
    files = os.listdir(folder)
    for file in files:
        with open(folder + file) as f:
            subject = f.readline().rstrip().split(' ')
            subject = list(map(int, subject[1:]))
            f.readline()
            data = list(map(int, f.readline().split(' ')))
            X.append(subject + data)
            if re.search(r'legit', file):
                y = np.append(y, int(1))
            else:
                y = np.append(y, int(0))
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
    # print(len(X_test))
    count = 0
    for i in range(len(X_test)):
        if X_test[i] > 0:
            p *= prob[y][i]
            count += 1
        else:
            p *= (1 - prob[y][i])
    return p


def predict(y_train, X_test, l, prob):
    # print(l[0], priori_prob(y_train, 0), likelihood_prob(X_test, 0, prob))
    # print(l[1], priori_prob(y_train, 1), likelihood_prob(X_test, 1, prob))
    y0 = l[0] * priori_prob(y_train, 0) * likelihood_prob(X_test, 0, prob)
    y1 = l[1] * priori_prob(y_train, 1) * likelihood_prob(X_test, 1, prob)
    if y0 > y1:
        return 0
    return 1


def cross_validation(X, y, n, l, alpha):
    acc = 0.0
    true_pred, false_pred, false_legit = 0, 0, 0
    k0, k1 = 0, 0
    for i in range(len(X)):
        k = 0
        ngrams, X_train = calc_ngrams(np.append(X[:i], X[i + 1:]), n)
        y_train = np.append((y[:i]), (y[i + 1:]))

        y_control = y[i]

        likelihood_y = calc_likelihood(X_train, y_train, alpha)
        # print(likelihood_y)

        y_test = []
        for j in range(len(X[i])):
            X_test = calc_X_test(ngrams, X[i][j], n)
            y_test.append(predict(y_train, X_test, l, likelihood_y))

        for j in range(len(y_control)):
            if y_test[j] == y_control[j]:
                k += 1
            if y_test[j] == 1 and y_control[j] == 1:
                # предсказано письмо и это письмо
                true_pred += 1
            if y_test[j] == 1 and y_control[j] == 0:
                # предсказано письмо, а это спам
                false_pred += 1
            if y_test[j] == 0 and y_control[j] == 1:
                # предсказан спам, а это письмо
                false_legit += 1

        # print(k / len(X[i]))
        acc += k / len(X[i])
        k0 += np.sum(y_control == 0)
        k1 += np.sum(y_control == 1)
    return acc / 10, false_legit, true_pred / k1, false_pred / k0


if __name__ == '__main__':
    X = []
    y = []
    dirs = os.listdir("./data/")
    for dir in dirs:
        xx, yy = read_data("./data/" + dir + "/")
        X.append(xx)
        y.append(yy)
    X = np.array(X)
    y = np.array(y)
    n = 1
    lambdaa = [i for i in range(1, int(1e308), int(1e308 / 10))]
    ox = []
    oy = []
    for l in lambdaa:
        acc, false_legit, true_pred, false_pred = cross_validation(X, y, n, (1, l), 0.001)
        print(np.log(float(l)), false_legit)

        ox.append(true_pred)
        oy.append(false_pred)

    plt.plot(ox, oy)
    plt.title("roc")
    plt.show()
