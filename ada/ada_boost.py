import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from utils.Suspects import Suspects


class AdaBoost(BaseEstimator):

    def __init__(self, estimator=None, estimator_n=100, suspect=None):
        self.estimator_n = estimator_n
        self.estimator = estimator
        self._estimators = np.array([])
        self.sample_weights = None
        self.stumps = None
        self.stump_weights = None
        self.suspect = suspect
        self._processed = 0

    def fit(self, X: np.ndarray, y: np.ndarray):

        n = X.shape[0]

        # init numpy arrays
        self.sample_weights = np.zeros(shape=(self.estimator_n, n))
        self.stumps = np.zeros(shape=self.estimator_n, dtype=object)
        self.stump_weights = np.zeros(shape=self.estimator_n)

        # initialize weights uniformly
        self.sample_weights[0] = np.ones(shape=n) / n

        for t in range(self.estimator_n):
            # fit  weak learner
            curr_sample_weights = self.sample_weights[t]
            stump = deepcopy(self.estimator)
            stump = stump.fit(X, y, sample_weight=curr_sample_weights)

            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
            err = curr_sample_weights[(stump_pred != y)].sum()  # / n
            stump_weight = np.log((1 - err) / err) / 2

            # update sample weights
            new_sample_weights = (
                    curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
            )

            new_sample_weights /= new_sample_weights.sum()

            # If not final iteration, update sample weights for t+1
            if t + 1 < self.estimator_n:
                self.sample_weights[t + 1] = new_sample_weights

            # save results of iteration
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self._processed += 1
            if self.suspect is not None:
                self.suspect.suspect(iteration=t, clf=self, error=err)

    def predict(self, X):
        built = self._processed
        sts = self.stumps[:built]
        ws = self.stump_weights[:built]

        print(built)
        print(self.stumps)
        print(sts)
        print(ws)
        stump_preds = np.array([stump.predict(X) for stump in sts])
        return np.sign(np.dot(self.stump_weights[:built], stump_preds))
