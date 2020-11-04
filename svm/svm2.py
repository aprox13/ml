import numpy as np


def calc_kenrel_matrix(kernel_func, A, B):
    n, *_ = A.shape
    m, *_ = B.shape
    f = lambda i, j: kernel_func(A[i], B[j])
    return np.fromfunction(np.vectorize(f), (n, m), dtype=int)


class SVM2:
    def __init__(self, kernel, C=1.0, max_iter=3000):
        self.kernef_f = kernel
        self.C = C

        self.b = 0

        self.EPS = 1e-6
        self.MAX_ITERATIONS = max_iter
        self.N = 0
        self.y = None
        self.kernel = None
        self.X = None
        self.alphas = np.zeros(0)

    def predict_soft(self, x):
        kernel = calc_kenrel_matrix(self.kernef_f, np.array([x]), self.X)
        return np.sum(self.alphas * self.y * kernel[0]) + self.b

    def predict(self, x):
        kernel = calc_kenrel_matrix(self.kernef_f, np.array([x]), self.X)
        res = int(np.sign(np.sum(self.alphas * self.y * kernel[0]) + self.b))
        return res if res != 0 else 1

    def get_random_j(self, i):
        res = np.random.randint(0, self.N - 1)
        return res if res < i else res + 1

    def calc_U_V(self, i, j):
        a_i, a_j = self.alphas[i], self.alphas[j]
        if self.y[i] == self.y[j]:
            U = max(0, a_i + a_j - self.C)
            V = min(self.C, a_i + a_j)
        else:
            U = max(0, a_j - a_i)
            V = min(self.C, self.C + a_j - a_i)
        return U, V

    def calc_E(self, i):
        return np.dot(self.alphas * self.y, self.kernel[i]) - self.y[i]

    def get_b(self, i):
        return 1 / self.y[i] - np.dot(self.alphas * self.y, self.kernel[i])

    def calc_b(self):
        self.b = 0
        idx = None
        for i in range(self.N):
            if self.EPS < self.alphas[i] and self.alphas[i] + self.EPS < self.C:
                idx = i
                break
        if idx is None:
            cnt = 0
            for i in range(self.N):
                if self.EPS < self.alphas[i]:
                    self.b += self.get_b(i)
                    cnt += 1
            if cnt != 0:
                self.b /= cnt
        else:
            self.b = self.get_b(idx)

    def fit(self, X, y):
        n, *_ = y.shape
        self.N = n
        self.y = y
        self.kernel = calc_kenrel_matrix(self.kernef_f, X, X)
        self.X = X
        self.alphas = np.zeros(n)

        indices = np.arange(self.N)
        for _ in range(self.MAX_ITERATIONS):
            np.random.shuffle(indices)
            for i_fake in range(self.N):
                i = indices[i_fake]
                j = indices[self.get_random_j(i_fake)]
                E_i = self.calc_E(i)
                E_j = self.calc_E(j)
                prev_a_i = self.alphas[i]
                prev_a_j = self.alphas[j]
                U, V = self.calc_U_V(i, j)
                if V - U < self.EPS:
                    continue
                eta = 2 * self.kernel[i][j] - (self.kernel[i][i] + self.kernel[j][j])
                if eta > -self.EPS:
                    continue
                possible_new_a_j = prev_a_j + self.y[j] * (E_j - E_i) / eta
                new_a_j = min(max(U, possible_new_a_j), V)
                if abs(new_a_j - prev_a_j) < self.EPS:
                    continue
                self.alphas[j] = new_a_j
                self.alphas[i] += self.y[i] * self.y[j] * (prev_a_j - new_a_j)
        self.calc_b()

    def get_support_indices(self):
        return np.where(np.logical_and(self.EPS < self.alphas,
                                       self.alphas + self.EPS < self.C))

    def get_bad_idx(self):
        return np.array([])
