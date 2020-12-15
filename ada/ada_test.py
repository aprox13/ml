from sklearn.model_selection import GridSearchCV
from tqdm.contrib.concurrent import process_map as pm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

from svm.smo import SMO
from utils.data_set import DataSet
from utils.methods import *
from utils.plots import *
from ada.ada_boost import AdaBoost
from sklearn.tree import DecisionTreeClassifier
from utils.Suspects import Suspects


class AdaSuspects(Suspects):
    def __init__(self, X, y, tX, ty, verbose=False):
        self._X = X
        self._y = y
        self._tX = tX
        self._ty = ty
        self.result_accuracy = {
            'test': [],
            'train': []
        }
        self.result_errors = []
        self.verbose = verbose
        self.iterations = 0

    def suspect(self, iteration=None, clf=None, error=None):
        assert iteration is not None
        assert clf is not None
        assert error is not None

        if self.verbose:
            print(f"Processed {iteration + 1} classifiers")

        def get_accuracy(X, y):
            y_true = y
            y_pred = clf.predict(X)
            if self.verbose:
                print(y_true)
                print(y_pred)
            return accuracy_score(y_true=y_true, y_pred=y_pred)

        self.result_accuracy['test'].append(get_accuracy(self._tX, self._ty))
        self.result_accuracy['train'].append(get_accuracy(self._X, self._y))
        self.result_errors.append(error)
        self.iterations += 1


def read_dataset(filename) -> DataSet:
    data = pd.read_csv(filename)
    X = data.values[:, :-1]
    tmp_y = data.values[:, -1]
    y = np.vectorize(lambda t: 1 if t == 'P' else -1)(tmp_y)
    return DataSet(X, y)


def process(data_set_name):
    filename = f"data/{data_set_name}.csv"
    ds = read_dataset(filename)
    X_train, X_test, y_train, y_test = train_test_split(ds.X, ds.y, test_size=0.33)

    suspect = AdaSuspects(X_train, y_train, X_test, y_test, verbose=True)

    ada = AdaBoost(estimator=DecisionTreeClassifier(), estimator_n=4, suspect=suspect)

    ada.fit(ds.X, ds.y)

    metric_plot(suspect.result_accuracy, list(range(suspect.iterations)), with_text=False)


if __name__ == '__main__':
    process("geyser")
