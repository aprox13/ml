import numpy as np
from sklearn.model_selection import train_test_split

from utils.data_set import DataSet
from utils.plots import *
from ada.ada_boost import AdaBoost
from sklearn.tree import DecisionTreeClassifier


def read_dataset(filename) -> DataSet:
    data = pd.read_csv(filename)
    X = data.values[:, :-1]
    tmp_y = data.values[:, -1]
    y = np.vectorize(lambda t: 1 if t == 'P' else -1)(tmp_y)
    return DataSet(X, y)


def make_grid(ds: DataSet, step=0.01):
    min_x, min_y = ds.X.min(axis=0)
    max_x, max_y = ds.X.max(axis=0)
    xx, yy = np.meshgrid(np.arange(min_x, max_x, step),
                         np.arange(min_y, max_y, step))

    grid = np.c_[xx.ravel(), yy.ravel()]
    print("xx shape", xx.shape)
    print("yy shape", yy.shape)
    print("grid shape", grid.shape)
    return xx, yy, grid


def draw_background(ds: DataSet, clf, xx, yy, grid):
    predict = clf.predict(grid).reshape(xx.shape)
    print(predict.ravel())
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(xx, yy, predict, cmap=plt.get_cmap('seismic'), shading='auto')
    # plt.scatter(x0, y0, color='red', s=100)
    # plt.scatter(x1, y1, color='blue', s=100)
    #
    # plt.scatter(x_sup, y_sup, color='white', marker='x', s=60)
    plt.show()


def draw(X, y, background_X, background_Y, title):
    positive_xs = [X[i][0] for i in range(len(X)) if y[i] >= 0]
    positive_ys = [X[i][1] for i in range(len(X)) if y[i] >= 0]

    negative_xs = [X[i][0] for i in range(len(X)) if y[i] < 0]
    negative_ys = [X[i][1] for i in range(len(X)) if y[i] < 0]

    pos_back_xs = [background_X[i][0] for i in range(len(background_X)) if background_Y[i] >= 0]
    pos_back_ys = [background_X[i][1] for i in range(len(background_X)) if background_Y[i] >= 0]

    neg_back_xs = [background_X[i][0] for i in range(len(background_X)) if background_Y[i] < 0]
    neg_back_ys = [background_X[i][1] for i in range(len(background_X)) if background_Y[i] < 0]

    plt.scatter(pos_back_xs, pos_back_ys, marker='.', color='green', alpha=0.2)
    plt.scatter(neg_back_xs, neg_back_ys, marker='.', color='red', alpha=0.2)
    plt.scatter(positive_xs, positive_ys, marker='+', color='green')
    plt.scatter(negative_xs, negative_ys, marker='_', color='red')

    plt.title(title)
    plt.show()


def print_shape(x):
    print(x.shape)


def callback(ds: DataSet, xx, yy, grid):
    def inner(clf=None, step=None):
        print(f"Clf {clf}, step {step}")
        pred = clf.predict(grid)
        print_shape(ds.X)
        print_shape(ds.y)
        print_shape(grid)
        print_shape(pred)

        draw(ds.X, ds.y, grid, pred, f"On step {step}")

    return inner


def process(data_set_name):
    filename = f"data/{data_set_name}.csv"
    ds = read_dataset(filename)

    xx, yy, grid = make_grid(ds)
    ada = AdaBoost(estimator=DecisionTreeClassifier(max_depth=2), estimator_n=4, callback=callback(ds, xx, yy, grid))

    ada.fit(ds.X, ds.y)


if __name__ == '__main__':
    process("geyser")
