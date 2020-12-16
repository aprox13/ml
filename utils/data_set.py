import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def data_set_from_csv(train_file: str, test_file: str = None):
    def read_file(file):
        ds = pd.read_csv(file)
        return ds.values[:, :-1], ds.values[:, -1]

    train_X, train_y = read_file(train_file)
    test_X, test_y = None, None

    if test_file is not None:
        test_X, test_y = read_file(test_file)
    return DataSet(train_X, train_y, test_X, test_y)


class DataSet:
    def __init__(self, X: np.ndarray, y: np.ndarray, test_X=None, test_y=None):
        self.X = X
        self.y = y

        self._test_X = test_X
        self._test_y = test_y

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

    def test_train_split(self, **kwargs):
        X, tX, y, ty = train_test_split(self.X, self.y, **kwargs)
        return DataSet(X, y), DataSet(tX, ty)

    def __repr__(self):
        count, features = self.X.shape
        test_count = self.get_test_X().shape[0] if self.get_test_X() is not None else 0
        return f"DataSet[features={features},count={count},test_count={test_count}]"


class DSWithSplit:

    def __init__(self, X, y, split):
        assert X is not None
        assert y is not None
        assert split is not None

        self.X = X
        self.y = y
        self.splits = split

    def __repr__(self):
        return f'DataSet[n_features={self.X.shape[1]},size={self.X.shape[0]},split_size={len(self.splits)}]'

    def split_first(self):
        assert len(self.splits) > 0

        splt = self.splits[0]
        return self.X[splt[0]], self.y[splt[0]], self.X[splt[1]], self.y[splt[1]]


class DSBuilder:
    _TEST = 'test'
    _TRAIN = 'train'

    def __init__(self):
        self._parts = []

    @staticmethod
    def of(ds: DataSet):
        b = DSBuilder()
        b.append(ds=ds)
        return b.build()

    def append(self, train_X: np.ndarray = None, train_y: np.ndarray = None,
               test_X: np.ndarray = None, test_y: np.ndarray = None,
               ds: DataSet = None):

        if ds is not None:
            return self.append(ds.get_X(), ds.get_y(), ds.get_test_X(), ds.get_test_y())

        check = {
            "train_X": train_X is None,
            "train_y": train_y is None,
            "test_X": test_X is None,
            "test_y": test_y is None
        }
        is_none = list(map(lambda kv: kv[0], filter(lambda kv: kv[1], check.items())))

        if len(is_none) != 0:
            raise RuntimeError("Couldn't append, because " + ", ".join(is_none) + " is none")

        if test_X.shape[0] != test_y.shape[0]:
            raise RuntimeError("Couldn't append, because test data have diff shape")

        if train_X.shape[0] != train_y.shape[0]:
            raise RuntimeError("Couldn't append, because train data have diff shape")

        self._parts.append(
            (train_X, train_y)
        )
        self._parts.append(
            (test_X, test_y)
        )
        return self

    def build(self) -> DSWithSplit:
        first_train, first_test = self._parts[0], self._parts[1]

        def concat(trn, tst):
            return np.concatenate((trn[0], tst[0])), np.concatenate((trn[1], tst[1]))

        res_X, res_y = concat(first_train, first_test)

        cv_res = [(0, len(first_train[0]), len(res_X))]

        for i in range(2, len(self._parts), 2):
            train = self._parts[i]
            test = self._parts[i + 1]
            start_idx = cv_res[-1][-1]

            add_X, add_y = concat(train, test)

            res_X = np.concatenate((res_X, add_X))
            res_y = np.concatenate((res_y, add_y))

            cv_res.append((start_idx, len(train[0]), len(res_X)))

        cv = []
        for start, len_train, end in cv_res:
            cv.append((range(start, start + len_train), range(start + len_train, end)))

        return DSWithSplit(res_X, res_y, list(map(lambda c: (list(c[0]), list(c[1])), cv)))
