from sklearn.model_selection import GridSearchCV
from tqdm.contrib.concurrent import process_map as pm

from svm.smo import SMO
from utils.data_set import DataSet
from utils.methods import *
from utils.plots import *

FILE_MASK = "data/{0}.csv"

CHOOSE_BEST_THREADS = 12


def read_dataset(filename) -> DataSet:
    data = pd.read_csv(filename)
    X = data.values[:, :-1]
    tmp_y = data.values[:, -1]
    y = np.vectorize(lambda t: 1 if t == 'P' else -1)(tmp_y)
    return DataSet(X, y)


def pm_predict(f, data: np.ndarray, name):
    return pm(f, data, desc=f'{name}', max_workers=CHOOSE_BEST_THREADS, chunksize=25000)


def draw(clf, ds: DataSet, step):
    X = ds.get_X()
    y = ds.get_y()
    x_min, y_min = np.amin(X, 0)
    x_max, y_max = np.amax(X, 0)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    mesh_dots = np.c_[xx.ravel(), yy.ravel()]

    predict_z = np.array(pm_predict(clf.predict_single, mesh_dots, name='predict')).reshape(xx.shape)
    # predict_z = np.array(np.apply_along_axis(clf.predict_single, 1, mesh_dots)).reshape(xx.shape)

    x0, y0 = X[y == -1].T
    x1, y1 = X[y == 1].T

    X_sup = X[clf.support_indices]
    x_sup, y_sup = X_sup.T

    def plot(_predict_z):
        plt.figure(figsize=(10, 10))
        plt.pcolormesh(xx, yy, _predict_z, cmap=plt.get_cmap('seismic'), shading='auto')
        plt.scatter(x0, y0, color='red', s=100)
        plt.scatter(x1, y1, color='blue', s=100)

        plt.scatter(x_sup, y_sup, color='white', marker='x', s=60)
        plt.show()

    plot(predict_z)


def flatten(a):
    if isinstance(a, list):
        if len(a) == 0:
            return []
        return flatten(a[0]) + flatten(a[1:])
    return [a]


BASIC_GRID = {
    "C": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0],
    'max_iters': [500, 1000]
}


def basic_with(d: dict):
    res = {}
    for k, v in BASIC_GRID.items():
        res[k] = v
    for k, v in d.items():
        res[k] = v
    return res


LINEAR_GRID = basic_with({'kernel': ['linear']})
RBF_GRID = basic_with({'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0]})
POLY_GRID = basic_with({'kernel': ['poly'], 'degree': [2.0, 3.0, 4.0, 5.0]})

GRID = [
    LINEAR_GRID,
    RBF_GRID,
    POLY_GRID
]


def choose_best(ds: DataSet):
    gs = GridSearchCV(estimator=SMO(),
                      param_grid=GRID,
                      cv=4,
                      scoring='accuracy',
                      verbose=1,
                      n_jobs=-1)

    gs.fit(ds.get_X(), ds.get_y())
    draw_metrics(gs.cv_results_)
    print(f'Got best score {gs.best_score_} with params {gs.best_params_}')
    return gs.best_params_


def kernel_to_s(d):
    kernel = d['kernel']
    if kernel == 'linear':
        return 'linear'
    if kernel == 'rbf':
        return f'rbf, γ={d["gamma"]}'
    if kernel == 'poly':
        return f'poly, d={d["degree"]}'
    return 'unknown'

metrics_params = {
    'linear': ['kernel'],
    'rbf': ['gamma']
}

def draw_metrics(cv_res):
    for iter in BASIC_GRID["max_iters"]:
        data = {}

        def append_metric(key, v):
            if key not in data:
                data[key] = []
            data[key].append(v)

        for C in BASIC_GRID['C']:
            for kernel in map(lambda d: d['kernel'][0], GRID):
                idxs = indices_where(has_all(C=C, max_iters=iter, kernel=kernel), cv_res['params'])
                for idx in idxs:
                    name = kernel_to_s(cv_res['params'][idx])
                    append_metric(name, cv_res['mean_test_score'][idx])
        metric_plot(
            data,
            BASIC_GRID['C'],
            f"Test score with max iters {iter}",
            x_label='C',
            with_text=False,
            default_color=True,
            fit_x=True,
            n_col=4
        )


def process(name):
    ds = log_action("Reading", lambda: read_dataset(FILE_MASK.format(name)), with_start_msg=True)
    svm_best_params = log_action("Choosing best svm", lambda: choose_best(ds), with_start_msg=True)
    svm_best = SMO()
    svm_best.set_params(**svm_best_params)

    print(f"Got {svm_best}")
    log_action("trainig", lambda: svm_best.fit(ds.get_X(), ds.get_y()), with_start_msg=True)
    y_pred = svm_best.predict(ds.get_X())

    from sklearn.metrics import accuracy_score
    sc = accuracy_score(ds.get_y(), y_pred.T)

    print(f'Train score {sc}')

    log_action("drawing", lambda: draw(svm_best, ds, step=0.01))


if __name__ == '__main__':
    process('chips')
