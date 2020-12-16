import numpy as np
from sklearn.base import BaseEstimator
from copy import deepcopy
import random


class AdaBoost(BaseEstimator):

    def __init__(self, estimator=None, estimator_n=100, callback=None):
        self.estimator_n = estimator_n
        self.estimator = estimator

        self.estimators = []
        self.alphas = []
        self.callback = callback

    def fit_one(self, X, y, weights):
        classifier = deepcopy(self.estimator)
        indices = random.choices(range(len(X)), weights=weights, k=len(X))
        new_X = [X[i] for i in indices]
        new_y = [y[i] for i in indices]
        classifier.fit(new_X, new_y)

        predicted = classifier.predict(X)
        error = sum([weights[i] for i in range(len(X)) if predicted[i] != y[i]])
        alpha = 0.5 * np.log((1 - error) / error) if np.isclose(0, error) else 1

        z = 0
        for i in range(len(weights)):
            weights[i] *= np.exp(-alpha * y[i] * predicted[i])
            z += weights[i]

        self.estimators.append(classifier)
        self.alphas.append(alpha)

        return weights / z

    def fit(self, X, y):
        weights = np.ones(len(X)) / len(X)
        for i in range(self.estimator_n):
            weights = self.fit_one(X, y, weights)
            if self.callback is not None:
                self.callback(clf=self, step=i + 1)

    def predict(self, X):
        return np.sign(sum([self.alphas[i] * self.estimators[i].predict(X) for i in range(len(self.estimators))]))

    def __repr__(self, N_CHAR_MAX=700):
        return f"AdaBoost(estimator={self.estimator},estimator_n={self.estimator_n}, fitted_count={len(self.estimators)})"
