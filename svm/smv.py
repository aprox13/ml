# coding:utf-8
import logging

import numpy as np
import scipy.spatial.distance as dist

np.random.seed(9999)


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
    def __init__(self, d=2):
        self.degree = d

    def __call__(self, x, y):
        return np.dot(x, y.T) ** self.degree

    def __repr__(self):
        return f"Poly[d = {self.degree}]"


class SVM:
    def __init__(self, C=1.0, kernel=None, tol=1e-6, max_iter=100):
        """Support vector machines implementation using simplified SMO optimization.
        Parameters
        ----------
        C : float, default 1.0
        kernel : Kernel object
        tol : float , default 1e-6
        max_iter : int, default 100
        """
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        if kernel is None:
            self.kernel = Linear()
        else:
            self.kernel = kernel

        self.b = 0
        self.alpha = None
        self.K = None
        self.X = None
        self.y = None
        self.n_samples = 0
        self.sv_idx = None

    def fit(self, X, y=None):
        self.n_samples = X.shape[0]
        self.X = X
        self.y = y
        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            self.K[:, i] = self.kernel(self.X, self.X[i, :])
        self.alpha = np.zeros(self.n_samples)
        self.sv_idx = np.arange(0, self.n_samples)
        return self._train()

    def _train(self):
        iters = 0
        while iters < self.max_iter:
            iters += 1
            alpha_prev = np.copy(self.alpha)

            for j in range(self.n_samples):
                # Pick random i
                i = self.random_index(j)

                eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                if eta >= 0:
                    continue
                L, H = self._find_bounds(i, j)

                # Error for current examples
                e_i, e_j = self._error(i), self._error(j)

                # Save old alphas
                alpha_io, alpha_jo = self.alpha[i], self.alpha[j]

                # Update alpha
                self.alpha[j] -= (self.y[j] * (e_i - e_j)) / eta
                self.alpha[j] = self.clip(self.alpha[j], H, L)

                self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (alpha_jo - self.alpha[j])

                # Find intercept
                b1 = (
                        self.b - e_i - self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, i]
                        - self.y[j] * (self.alpha[j] - alpha_jo) * self.K[i, j]
                )
                b2 = (
                        self.b - e_j - self.y[j] * (self.alpha[j] - alpha_jo) * self.K[j, j]
                        - self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, j]
                )
                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = 0.5 * (b1 + b2)

            # Check convergence
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break
        logging.info("Convergence has reached after %s." % iters)

        # Save support vectors index
        self.sv_idx = np.where(self.alpha > 0)[0]

    def predict(self, X):
        return np.sign(self._predict_row(X))

    def predict_soft(self, X):
        return self._predict_row(X)

    def _predict_row(self, X):
        k_v = self.kernel(self.X[self.sv_idx], X)
        return np.dot((self.alpha[self.sv_idx] * self.y[self.sv_idx]).T, k_v.T) + self.b

    def clip(self, alpha, H, L):
        if alpha > H:
            alpha = H
        if alpha < L:
            alpha = L
        return alpha

    def _error(self, i):
        """Error for single example."""
        return self._predict_row(self.X[i]) - self.y[i]

    def _find_bounds(self, i, j):
        """Find L and H such that L <= alpha <= H.
        Also, alpha must satisfy the constraint 0 <= αlpha <= C.
        """
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H

    def random_index(self, z):
        i = z
        while i == z:
            i = np.random.randint(0, self.n_samples - 1)
        return i

    def get_support_indices(self):
        return self.sv_idx

    def __repr__(self):
        return f"SVM[kernel={self.kernel}, C={self.C}]"

    def get_bad_idx(self):
        return np.where(self.alpha == self.C)[0]

    def stat(self):
        def select(f):
            return self.X[np.where(f)[0]].shape[0]

        print(f"""
        a < 0: {select(self.alpha < 0)}
        a = 0: {select(self.alpha == 0)}
        a 0-C: {select((self.alpha > 0) & (self.alpha < self.C))}
        a = C: {select(self.alpha == self.C)} 
        a > C: {select(self.alpha > self.C)}
        """)
