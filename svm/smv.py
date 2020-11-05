import numpy as np
import scipy.spatial.distance as dist

np.random.seed(1982)
from sklearn.base import BaseEstimator


class Linear(object):
    def __call__(self, x, y):
        return np.dot(x, y.T)

    def __repr__(self):
        return "Linear"


class RBF(object):
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        return np.exp(-self.beta * dist.cdist(x, y) ** 2).flatten()

    def __repr__(self):
        return f"RBF[β = {self.beta}]"


class Poly(object):
    def __init__(self, d=2, k=1):
        self.degree = d
        self.k = k

    def __call__(self, x, y):
        return np.dot(x, y.T) ** self.degree + self.k

    def __repr__(self):
        return f"Poly[d = {self.degree}, k = {self.k}]"


class SVM(BaseEstimator):
    def __init__(self, C=1.0, kernel=None, eps=1e-6, max_iter=100):
        self.C = C
        self.eps = eps
        self.max_iter = max_iter

        self.kernel = kernel

        self.b = 0
        self.lagr_coefs = None
        self.K = None
        self.X = None
        self.y = None
        self.n_samples = 0
        self.support_indices = None

    def fit(self, X, y=None):
        self.n_samples = X.shape[0]
        self.X = X
        self.y = y
        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            self.K[:, i] = self.kernel(self.X, self.X[i, :])
        self.lagr_coefs = np.zeros(self.n_samples)
        self.support_indices = np.arange(0, self.n_samples)

        for _ in range(self.max_iter):
            prev_coefs = np.copy(self.lagr_coefs)

            for j in range(self.n_samples):
                i = j
                while i == j:
                    i = np.random.randint(0, self.n_samples - 1)

                eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                if eta < 0:
                    if self.y[i] == self.y[j]:
                        L = max(0, self.lagr_coefs[i] + self.lagr_coefs[j] - self.C)
                        H = min(self.C, self.lagr_coefs[i] + self.lagr_coefs[j])
                    else:
                        L = max(0, self.lagr_coefs[j] - self.lagr_coefs[i])
                        H = min(self.C, self.C - self.lagr_coefs[i] + self.lagr_coefs[j])

                    e_i = self._predict_one(self.X[i]) - self.y[i]
                    e_j = self._predict_one(self.X[j]) - self.y[j]

                    old_c_i, old_c_j = self.lagr_coefs[i], self.lagr_coefs[j]

                    self.lagr_coefs[j] -= (self.y[j] * (e_i - e_j)) / eta
                    self.lagr_coefs[j] = max(min(self.lagr_coefs[j], H), L)

                    self.lagr_coefs[i] = self.lagr_coefs[i] + self.y[i] * self.y[j] * (old_c_j - self.lagr_coefs[j])

                    def b(a, b, e_a, old_a, old_b):
                        return self.b - e_a - self.y[a] * (self.lagr_coefs[a] - old_a) * self.K[a, a] \
                               - self.y[b] * (self.lagr_coefs[b] - old_b) * self.K[i, j]

                    self.b = b(i, j, e_i, old_c_i, old_c_j)
                    if not (0 < self.b < self.C):
                        p = self.b
                        self.b = b(j, i, e_j, old_c_j, old_c_i)
                        if not (0 < self.b < self.C):
                            self.b = 0.5 * (p + self.b)

            if np.linalg.norm(self.lagr_coefs - prev_coefs) < self.eps:
                break

        self.support_indices = np.where(self.lagr_coefs > 0)[0]

    def predict(self, X):
        res = []
        for row in X:
            cls = np.sign(self._predict_one(row))
            res.append(1 if cls == 0 else cls)
        return np.array(res)

    def predict_single(self, X):
        return np.sign(self._predict_one(X))

    def _predict_one(self, X):
        k_v = self.kernel(self.X[self.support_indices], X)
        return np.dot((self.lagr_coefs[self.support_indices] * self.y[self.support_indices]).T, k_v.T) + self.b

    def __repr__(self, **kwargs):
        return f"SVM[kernel={self.kernel}, C={self.C}]"
