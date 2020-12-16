import random
from sklearn.metrics import accuracy_score
import imageio
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from functools import reduce
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

from utils.plots import metric_plot
from utils.data_set import DataSet

STEPS = 30
GRID_POINTS = 100
GIF_FPS = 3


def read_dataset(filename) -> DataSet:
    data = pd.read_csv(filename)
    X = data.values[:, :-1]
    tmp_y = data.values[:, -1]
    y = np.vectorize(lambda t: 1 if t == 'P' else -1)(tmp_y)
    return DataSet(X, y)


def generate_gif(name):
    gn = f'img/{name}/{random.randint(0, 100000000)}.gif'
    imageio.mimsave(
        gn,
        [imageio.imread(f"img/{name}/{i}.png") for i in range(1, STEPS)],
        fps=GIF_FPS
    )
    print(f'![SegmentLocal]({gn} "{name}")')


def initial_weights(n):
    return np.ones(n) / n


class AdaBoost:
    def __init__(self, n_estimator=100, callback=None, verbose=False):
        self.clfs = []
        self.n_estimator = n_estimator
        self.a = np.array([])
        self.callback = callback
        self.verbose = verbose

    def fit_one(self, X, y, weights, step):
        classifier = DecisionTreeClassifier(max_depth=2)
        indices = random.choices(range(len(X)), weights=weights, k=len(X))
        classifier.fit(X[indices], y[indices])

        predicted = classifier.predict(X)
        error = weights[predicted != y].sum()

        alpha = 0.5 * np.log((1 - error) / error) if np.isclose(0, error) else 1

        weights *= np.exp(np.multiply(y, predicted) * (-alpha))

        self.clfs.append(classifier)
        self.a = np.append(self.a, alpha)
        if self.callback is not None:
            self.callback(clf=self, step=step + 1)
        return weights / weights.sum()

    def fit(self, X, y):
        reduce(
            lambda ws, step: self.fit_one(X, y, ws, step),
            tqdm(range(self.n_estimator)) if self.verbose else range(self.n_estimator),
            initial_weights(len(X))
        )

    def predict(self, X):
        pred = np.array(list(map(lambda clf: clf.predict(X), self.clfs)))
        return np.sign(self.a @ pred)


def scatter(X, **kwargs):
    assert X.shape[1] == 2

    xx = X[:, 0]
    yy = X[:, 1]
    plt.scatter(xx, yy, **kwargs)


def draw(ds: DataSet, bgX, bgY, step, name):
    X = ds.X
    y = ds.y
    fig = plt.figure()

    scatter(X[y >= 0], marker='+', color='green')
    scatter(X[y < 0], marker='_', color='red')
    scatter(bgX[bgY >= 0], marker='.', color='green', alpha=0.15)
    scatter(bgX[bgY < 0], marker='.', color='red', alpha=0.15)

    plt.title(f'step {step}/{STEPS}')
    fig.savefig(f"img/{name}/{step}")
    plt.close(fig)


def callback(ds: DataSet, bgX, name):
    def inner(clf=None, step=None):
        bgY = clf.predict(bgX)
        draw(ds, bgX, bgY, step, name)

    return inner


def train(ds: DataSet, name):
    print("Train")
    min_x, min_y = ds.X.min(axis=0)
    max_x, max_y = ds.X.max(axis=0)

    background_X = np.array(
        list(
            zip(
                np.repeat(np.linspace(min_x, max_x, GRID_POINTS), GRID_POINTS),
                np.tile(np.linspace(min_y, max_y, GRID_POINTS), GRID_POINTS)
            )
        )
    )

    clf = AdaBoost(n_estimator=STEPS, callback=callback(ds, background_X, name), verbose=True)
    clf.fit(ds.X, ds.y)


def test(ds: DataSet, name):
    print("Metric")
    train_ds, test_ds = ds.test_train_split(test_size=0.33)

    metric_data = {
        "test": [],
        "train": []
    }

    def add_metric(ds, clf, ds_name):
        metric_data[ds_name].append(accuracy_score(ds.y, clf.predict(ds.X)))

    def clbck(clf, step):
        add_metric(test_ds, clf, "test")
        add_metric(train_ds, clf, "train")

    clf = AdaBoost(n_estimator=STEPS, callback=clbck, verbose=True)
    clf.fit(train_ds.X, train_ds.y)

    metric_plot(metric_data, x_label='Steps', x_values=list(range(1, STEPS + 1)), title=f'Accuracy for {name}',
                default_color=True)


def safe(f):
    try:
        f()
    except:
        pass


def process(name):
    safe(os.makedirs(f"img/{name}"))

    ds = read_dataset(f'data/{name}.csv')

    train(ds, name)
    test(ds, name)
    generate_gif(name)


process("chips")
