import numpy as np
from sklearn.base import BaseEstimator
import scipy.spatial.distance as dist


def random(not_eq, max_ex, min_inc=0):
    itr = 0
    while itr < 1000:
        j = np.random.randint(min_inc, max_ex - 1)
        if j != not_eq:
            return j
        itr += 1
    raise RuntimeError('Couldn\'t get random')


def bound(value, min_v, max_v):
    return max(min_v, min(max_v, value))


def join_funcs(f1, f2):
    def inner(*args, **kwargs):
        return f1(f2(*args, **kwargs))

    return inner


class SMO(BaseEstimator):

    def _kernel_linear(self, x, y):
        if x.shape == y.shape:
            return np.dot(x, y)
        return np.dot(x, y.T)

    def _kernel_poly(self, x, y):
        if x.shape == y.shape:
            return np.dot(x, y) ** self.degree + 1
        return np.dot(x, y.T) ** self.degree + 1

    def _kernel_rbf(self, x, y):
        return np.exp(-self.gamma * dist.cdist(x, y) ** 2).flatten()

    def _kernel(self, x, y):
        return SMO.__getattribute__(self, f'_kernel_{self.kernel}')(x, y)

    def __init__(self, eps=0.000001, kernel='linear', gamma=0, degree=0, C=1, max_iters=1000):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.eps = eps
        self.max_iters = max_iters

        self.a = None
        self.b = 0
        self.X = np.array([])
        self.y = np.array([])
        self._kernel_matrix = np.array([])
        self.support_indices = np.array([])
        self.X_for_predict = np.array([])
        self.a_dot_y = np.array([])
        self.degree = degree

    def fit(self, X, y):
        self.X = X
        self.y = y
        C = self.C
        N = len(X)
        self._kernel_matrix = np.array(
            [self._kernel(X[i], X[j])
             for i in range(N)
             for j in range(N)
             ]
        ).reshape((N, N))

        self.a = np.zeros(N)
        for _ in range(self.max_iters):
            alpha_prev = np.copy(self.a)

            for j in range(N):
                i = random(not_eq=j, max_ex=N)

                K_ii = self._kernel_matrix[i, i]
                K_ij = self._kernel_matrix[i, j]
                K_jj = self._kernel_matrix[j, j]

                y_i = self.y[i]
                y_j = self.y[j]

                a_i = self.a[i]
                a_j = self.a[j]

                eta = 2.0 * K_ij - K_ii - K_jj
                if eta >= 0:
                    continue

                if y_i == y_j:
                    L = max(0, a_i + a_j - C)
                    H = min(C, a_i + a_j)
                else:
                    L = max(0, a_j - a_i)
                    H = min(C, C - a_i + a_j)

                E_i = self._predict_learn(i) - self.y[i]
                E_j = self._predict_learn(j) - self.y[j]

                self.a[j] = bound(
                    a_j - (y_j * (E_i - E_j)) / eta,
                    max_v=H,
                    min_v=L)

                self.a[i] = a_i + y_i * y_j * (a_j - self.a[j])

                da_i = self.a[i] - a_i
                da_j = self.a[j] - a_j

                b1 = self.b - E_i - y_i * da_i * K_ii - y_j * da_j * K_ij
                b2 = self.b - E_j - y_j * da_j * K_jj - y_i * da_i * K_ij

                if 0 < self.a[i] < C:
                    self.b = b1
                elif 0 < self.a[j] < C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2.0

            if np.linalg.norm(self.a - alpha_prev) < self.eps:
                break

        self.support_indices = np.where(self.a > 0)[0]

        self.X_for_predict = self.X[self.support_indices]
        self.a_dot_y = (self.a[self.support_indices] * self.y[self.support_indices]).T

    def predict(self, X):
        return np.apply_along_axis(self.predict_single, 1, X)

    def _predict_learn(self, i):
        k_v = self._kernel_matrix[i, :]  # X @ X[i]
        return np.dot((self.a * self.y), k_v) + self.b

    def predict_single(self, X):
        k_v = self._kernel(self.X_for_predict, X)
        return np.sign(np.dot(self.a_dot_y, k_v) + self.b)
