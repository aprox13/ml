import math

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class DecisionForest(BaseEstimator):

    @staticmethod
    def _cnt_all(n):
        return n

    @staticmethod
    def _cnt_sqrt(n):
        return math.ceil(math.sqrt(n))

    def _get_count(self, name, n):
        return DecisionForest.__getattribute__(self, f'_cnt_{name}')(n)

    def __init__(self, trees_cnt=1, samples_per_tree='all', features_per_tree='all', tree_params=None):
        if tree_params is None:
            tree_params = {}
        self.tree_params = tree_params
        self.trees_cnt = trees_cnt
        self.samples_per_tree = samples_per_tree
        self.features_per_tree = features_per_tree
        self.trees_features = None
        self.trees = None

    def train_tree(self, idx, X, y):
        cnt, dim = X.shape
        tree_sample_size = self._get_count(self.samples_per_tree, cnt)
        selected = np.random.choice(cnt, tree_sample_size, replace=True)
        tree = DecisionTreeClassifier()
        tree.set_params(**self.tree_params)
        tree.fit(X[selected][:, self.trees_features[idx]], y[selected])
        return tree

    def fit(self, X, y):
        _, M = X.shape
        tree_features = self._get_count(self.features_per_tree, M)
        self.tree_params['max_features'] = tree_features
        self.trees_features = [np.random.choice(M, tree_features, replace=False) for _ in range(self.trees_cnt)]
        self.trees = Parallel(n_jobs=-1)(
            delayed(self.train_tree)(i, X, y)
            for i in range(self.trees_cnt)
        )

    def predict_one(self, x):
        decisions = np.zeros(self.trees_cnt)

        def tree_pred_one(i, x):
            xx = [[]]
            for k in x:
                xx[0].append(k)
            return self.trees[i].predict(np.array(xx))[0]

        for i in range(self.trees_cnt):
            decisions[i] = tree_pred_one(i, x[self.trees_features[i]])

        groups, cnt = np.unique(decisions, return_counts=True)
        return groups[np.argmax(cnt)]

    def predict(self, X):
        return np.apply_along_axis(self.predict_one, 1, X)
