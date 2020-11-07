from typing import List

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from utils.data_set import *

FILES_COUNT = 21
MAX_DEPTH_LIMIT = 30
VERBOSE_GRID_SEARCH = 1


def csv_file(i, tp):
    return f'data/{i if i >= 10 else ("0" + str(i))}_{tp}.csv'


def test_file(i):
    return csv_file(i, 'test')


def train_file(i):
    return csv_file(i, 'train')


def data_sets() -> List[DSWithSplit]:
    dss = [data_set_from_csv(train_file(i), test_file(i)) for i in range(1, FILES_COUNT + 1)]

    def build(ds: DataSet) -> DSWithSplit:
        ds_builder = DSBuilder()
        ds_builder.append(ds.get_X(), ds.get_y(), ds.get_test_X(), ds.get_test_y())
        return ds_builder.build()

    return list(map(build, dss))


GRID = {
    'criterion': ["gini", "entropy"],
    'splitter': ["best", "random"],
    'max_depth': list(range(1, MAX_DEPTH_LIMIT + 1))
}


def optimize_params(ds: DSWithSplit) -> GridSearchCV:
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=GRID,
        n_jobs=-1,
        scoring='accuracy',
        return_train_score=True,
        verbose=VERBOSE_GRID_SEARCH,
        cv=ds.split
    )
    res = grid_search.fit(ds.X, ds.y)
    return res


def process_data_set(ds: DSWithSplit):
    print(f"Processing {ds}")
    cv_res = optimize_params(ds)
    # print(cv_res.cv_results_)

    d = {}
    for k, v in cv_res.cv_results_.items():
        d[k] = v[cv_res.best_index_]

    print(d)


if __name__ == '__main__':
    process_data_set(data_sets()[0])
