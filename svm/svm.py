from functools import partial
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import time
from tqdm.contrib.concurrent import process_map
from functools import reduce

FILE_MASK = "data/{0}.csv"
ITERATIONS = 3000

CHOOSE_BEST_THREADS = 7

def in_range(x, start_ex, end_ex):
    return start_ex < x < end_ex


def find_in(array, by):
    return next(filter(by, array), None)


def pretty_time(millis: int) -> str:
    base = [(1000 * 60, "min"), (1000, "sec"), (1, "ms")]

    def step(acc, x):
        cur_millis, result = acc
        multiplier, name = x

        part = cur_millis // multiplier
        if part != 0:
            result.append(f"{part}{name}")
            cur_millis -= part * multiplier
            return cur_millis, result
        return acc

    res = reduce(step, base, (millis, []))[1]
    if len(res) != 0:
        return "".join(res)
    return "0ms"


def log_action(action_name, action, with_start_msg=False, with_result=True):
    def millis():
        return int(round(time.time() * 1000))

    if with_start_msg:
        print(f"starting '{action_name}'")

    start = millis()
    res = action()
    end_time_s = pretty_time(millis() - start)
    result_part = ""
    if with_result:
        result_part = f" with result {res}"

    print(f"'{action_name}' ends in {end_time_s}{result_part}")
    return res


class DataSet:
    def __init__(self, X: np.ndarray, y: np.ndarray, test_X=None, test_y=None):
        self.X = X
        self.y = y

        self._test_X = test_X
        self._test_y = test_y

    def shuffle(self):
        indices = np.arange(self.y.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_test_X(self):
        return self._test_X

    def get_test_y(self):
        return self._test_y

    def count(self):
        return self.y.shape[0]

    def get_for_cross_validation(self, train_indices, test_indices):
        train_X, test_X = self.X[train_indices], self.X[test_indices]
        train_y, test_y = self.y[train_indices], self.y[test_indices]

        return DataSet(X=train_X, y=train_y, test_X=test_X, test_y=test_y)


def read_dataset(filename) -> DataSet:
    data = pd.read_csv(filename)
    X = data.values[:, :-1]
    tmp_y = data.values[:, -1]
    y = np.vectorize(lambda t: 1 if t == 'P' else -1)(tmp_y)
    return DataSet(X, y)


class Kernel:
    def __init__(self, name: str, C: float, gamma=None, k0=None, degree=None):
        self._name = name
        self._C = C
        self._gamma = gamma
        self._k0 = k0
        self._degree = degree

    def name(self):
        return self._name

    def _to_str(self):
        params = [
            ("Kernel", self._name),
            ("C", self._C),
            ("gamma", self._gamma),
            ("k0", self._k0),
            ("degree", self._degree)
        ]

        filtered = filter(lambda x: x[1] is not None, params)
        mapped = list(map(lambda x: f"{x[0]}: {x[1]}", filtered))

        res = ", ".join(mapped)

        return f"({res})"

    def __repr__(self):
        return self._to_str()

    def calculate(self, a, b):
        pass

    # noinspection PyTypeChecker
    def matrix(self, A, B) -> np.ndarray:
        def function(i, j):
            return self.calculate(A[i], B[j])

        shape = (A.shape[0], B.shape[0])

        return np.fromfunction(
            np.vectorize(function),
            shape,
            dtype=int
        )

    def get_C(self):
        return self._C


class LinearKernel(Kernel):
    def __init__(self, C: float):
        super().__init__('linear', C)

    def calculate(self, a, b):
        print(f"Linear on {a} and {b}")
        return np.dot(a, b)


class PolyKernel(Kernel):
    def __init__(self, C: float, gamma, k0, degree):
        super().__init__("poly", C, gamma, k0, degree)

    def calculate(self, a, b):
        return np.power(self._gamma * np.dot(a, b) + self._k0, self._degree)


class SigmodKernel(Kernel):
    def __init__(self, C: float, gamma, k0):
        super().__init__('sigmod', C, gamma, k0)

    def calculate(self, a, b):
        return np.tanh(self._gamma * np.dot(a, b) + self._k0)


def build_kernels(params: dict) -> List[Kernel]:
    def non_empty_or_stub(key):
        if key in params and len(params[key]) != 0:
            return params[key]
        return [{}]

    res = [
        params['creator'](C, gamma, k0, degree)
        for C in non_empty_or_stub("C")
        for gamma in non_empty_or_stub("gamma")
        for k0 in non_empty_or_stub("k0")
        for degree in non_empty_or_stub("degree")
    ]
    return res


class SVM:
    class E:
        def __init__(self, i, j):
            self.i = i
            self.j = j

    def __init__(self, data_set: DataSet, kernel: Kernel, eps=1e-6, iterations=4000):
        self.kernel = kernel
        self.kernel_matrix = kernel.matrix(data_set.get_X(), data_set.get_X())

        self.data_set = data_set
        self.EPS = eps
        self.iterations = iterations

        self._b = 0
        self._alphas = np.zeros(data_set.count())

    def _b_provider(self, i):
        y = self.data_set.get_y()

        return 1 / y[i] - np.dot(self._alphas * y, self.kernel_matrix[i])

    def _get_E(self, i, j):
        y = self.data_set.get_y()
        alphas = self._alphas

        def formula(idx):
            return np.dot(alphas * y, self.kernel_matrix[idx]) - y[idx]

        return self.E(formula(i), formula(j))

    def calc_U_V(self, i, j):
        y = self.data_set.get_y()
        C = self.kernel.get_C()

        a_i, a_j = self._alphas[i], self._alphas[j]
        if y[i] == y[j]:
            U = max(0, a_i + a_j - C)
            V = min(C, a_i + a_j)
        else:
            U = max(0, a_j - a_i)
            V = min(C, C + a_j - a_i)
        return U, V

    def _calculate_b(self):
        indices = range(self.data_set.count())

        idx = find_in(
            array=indices,
            by=lambda i: in_range(
                x=self._alphas[i],
                start_ex=self.EPS,
                end_ex=self.kernel.get_C() - self.EPS
            )
        )
        if idx is not None:
            self._b = self._b_provider(idx)
        else:
            suitable = list(
                map(
                    lambda x: self._b_provider(x),
                    filter(lambda x: self.EPS < self._alphas[x], indices)
                )
            )
            if len(suitable) != 0:
                self._b = sum(suitable) / len(suitable)
            else:
                self._b = 0

    def _iteration(self):
        indices = np.arange(self.data_set.count())
        np.random.shuffle(indices)

        def get_i_j(x, idxs):
            i_res = idxs[x]
            y = np.random.randint(0, self.data_set.count() - 1)
            j_res = idxs[y if y < x else y + 1]
            return i_res, j_res

        for it_step in range(self.data_set.count()):
            i, j = get_i_j(it_step, indices)
            e = self._get_E(i, j)
            prev_a_j = self._alphas[j]
            U, V = self.calc_U_V(i, j)

            kernel_m = self.kernel_matrix
            y = self.data_set.get_y()
            alphas = self._alphas

            if V - U < self.EPS:
                continue

            eta = 2 * kernel_m[i][j] - (kernel_m[i][i] + kernel_m[j][j])
            if eta == 0:
                continue
            possible_new_a_j = prev_a_j + y[j] * (e.j - e.i) / eta
            new_a_j = min(max(U, possible_new_a_j), V)

            if abs(new_a_j - prev_a_j) < self.EPS:
                continue
            alphas[j] = new_a_j
            alphas[i] += y[i] * y[j] * (prev_a_j - new_a_j)

    def fit(self):
        for _ in range(self.iterations):
            self._iteration()
        self._calculate_b()

    def predict(self, x):
        kernel_m = self.kernel.matrix(np.array([x]), self.data_set.get_X())
        y = self.data_set.get_y()
        res = int(np.sign(np.sum(self._alphas * y * kernel_m[0]) + self._b))
        return res if res != 0 else 1

    def get_suport_indices(self):
        return np.where(np.logical_and(self.EPS < self._alphas,
                                       self._alphas + self.EPS < self.kernel.get_C()))


def get_clf(kernel: Kernel, data_set: DataSet) -> SVM:
    clf = SVM(data_set=data_set, kernel=kernel, iterations=ITERATIONS)
    clf.fit()
    return clf


def get_score(kernel: Kernel, data_set: DataSet):
    cv = KFold(4)
    f_scores = []
    for train_index, test_index in cv.split(data_set.get_X()):
        cv_data_set = data_set.get_for_cross_validation(train_index, test_index)
        clf = get_clf(kernel, cv_data_set)
        y_pred = np.apply_along_axis(clf.predict, 1, cv_data_set.get_test_X())
        f_scores.append(f1_score(cv_data_set.get_test_y(), y_pred))
    return np.average(np.array(f_scores))


def choose_kernel(ds: DataSet, kernels) -> Tuple[Kernel, float]:
    scores = process_map(partial(get_score, data_set=ds), kernels, max_workers=CHOOSE_BEST_THREADS)

    return kernels[np.argmax(np.array(scores))]


def draw(clf, X, y, step):
    stepx = step
    stepy = 0.01
    x_min, y_min = np.amin(X, 0)
    x_max, y_max = np.amax(X, 0)
    x_min -= stepx
    x_max += stepx
    y_min -= stepy
    y_max += stepy
    xx, yy = np.meshgrid(np.arange(x_min, x_max, stepx),
                         np.arange(y_min, y_max, stepy))

    mesh_dots = np.c_[xx.ravel(), yy.ravel()]
    zz = np.apply_along_axis(lambda t: clf.predict(t), 1, mesh_dots)
    zz = np.array(zz).reshape(xx.shape)

    plt.figure(figsize=(10, 10))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    x0, y0 = X[y == -1].T
    x1, y1 = X[y == 1].T

    plt.pcolormesh(xx, yy, zz, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), shading='auto')
    plt.scatter(x0, y0, color='red', s=100)
    plt.scatter(x1, y1, color='blue', s=100)

    sup_ind = clf.get_suport_indices()
    X_sup = X[sup_ind]
    x_sup, y_sup = X_sup.T

    plt.scatter(x_sup, y_sup, color='white', marker='x', s=60)
    plt.show()


def get_best_kernel2(data_set: DataSet, builders):
    kernels_array = [
        kernel
        for params in builders
        for kernel in build_kernels(params)
    ]

    kernels = np.array(kernels_array).flatten()
    return choose_kernel(data_set, kernels)


def process(name: str, builders):
    file = FILE_MASK.format(name)
    data_set = read_dataset(file)
    data_set.shuffle()

    kernel = log_action("Getting best kernel", lambda: get_best_kernel2(data_set, builders), with_start_msg=False)

    clf = log_action(
        f"Fitting SVM for {kernel}",
        lambda: get_clf(kernel, data_set),
        with_start_msg=True,
        with_result=False
    )

    log_action(
        f"Drawing",
        lambda: draw(clf, data_set.get_X(), data_set.get_y(), 0.01),
        with_start_msg=True,
        with_result=False
    )


LINEAR = {
    "C": [0.1, 1.0, 10.0, 25.0, 50.0, 100.0],
    "creator": lambda C, g, k0, d: LinearKernel(C)
}

SIGMOD = {
    "C": [0.1, 1.0, 10.0, 25.0, 50.0, 100.0],
    "gamma": [0.5, 0.75, 1.0],
    "k0": [0.01, 0.5, 1.0],
    "creator": lambda C, g, k0, d: SigmodKernel(C, g, k0),
}

POLY = {
    "C": [0.1, 1.0, 10.0, 25.0, 50.0, 100.0],
    "gamma": [1.0],
    "k0": [1.0],
    "degree": [3, 4, 5],
    "creator": lambda C, g, k0, d: PolyKernel(C, g, k0, d)
}

BUILDERS = [
    LINEAR,
    SIGMOD,
    POLY
]


# process("geyser", BUILDERS)


# def make_args(d):
#     res = []
#     print(type(d))
#
#     indices = [
#         (0, key) for key in d
#     ]
#
#     base_index = 0
#
#     print(d)
#
#
# test = {
#     'a': [1, 2],
#     'b': 10,
#     'c': [5, 6]
# }
#
#
# def make_list(x):
#     if type(x) == list:
#         return x
#     return [x]
#
#
# def list_add(lst, x):
#     res = []
#     for l in lst:
#         t = []
#         for e in l:
#             t.append(e)
#         t.append(x)
#         res.append(t)
#
#     return res
#
#
# def rec(args: dict, res):
#     if len(args) == 0:
#         return res
#     key = list(args.keys())[0]
#     new_res = []
#     for v in make_list(args[key]):
#         new_res += list_add(res, (key, v))
#
#     del args[key]
#     return rec(args, new_res)
#
#
# print(rec(test, [[]]))
#
# """
# 1 10 5
# 1 10 6
# 2 10 5
# 2 10 6
# """
#

_X = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

X = np.array(_X)

for i in range(3):
    print(X[i, :])


print()
for i in range(3):
    print(X[:, i])


