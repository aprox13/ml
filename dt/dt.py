from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import copy

from utils.data_set import *
from utils.methods import *
from utils.plots import *

FILES_COUNT = 21
MAX_DEPTH_LIMIT = 30
VERBOSE_GRID_SEARCH = 1

DEPTH_CHOOSE = list(range(1, MAX_DEPTH_LIMIT + 1))
CRITERION = 'criterion'
SPLITTER = 'splitter'
GRID = {
    CRITERION: ["gini", "entropy"],
    SPLITTER: ["best", "random"],
    'max_depth': DEPTH_CHOOSE
}


def csv_file(i, tp):
    return f'data/{i if i >= 10 else ("0" + str(i))}_{tp}.csv'


def test_file(i):
    return csv_file(i, 'test')


def train_file(i):
    return csv_file(i, 'train')


def read_data_sets() -> List[DSWithSplit]:
    data_sets = [data_set_from_csv(train_file(i), test_file(i)) for i in range(1, FILES_COUNT + 1)]

    return list(map(DSBuilder.of, data_sets))


def optimize_params(ds: DSWithSplit, verbose: bool) -> GridSearchCV:
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=GRID,
        n_jobs=-1,
        scoring='accuracy',
        return_train_score=True,
        verbose=VERBOSE_GRID_SEARCH if verbose else 0,
        cv=ds.split
    )
    res = grid_search.fit(ds.X, ds.y)
    return res


PARAM_MAX_DEPTH = 'param_max_depth'
TRAIN_SCORE = 'split0_train_score'
TEST_SCORE = 'split0_test_score'
PARAMS = 'params'
MAX_DEPTH = 'max_depth'


def process_data_set(ds: DSWithSplit, verbose: bool):
    if verbose:
        print(f"Processing {ds}")
    cv_res = optimize_params(ds, verbose)
    params = cv_res.best_params_
    to_search = copy.deepcopy(params)

    statistic = {
        "test": [],
        "train": []
    }

    for i in range(len(DEPTH_CHOOSE)):
        to_search[MAX_DEPTH] = DEPTH_CHOOSE[i]
        idx = index_where(dict_contains(to_search), cv_res.cv_results_[PARAMS])

        statistic['test'].append(float(cv_res.cv_results_[TEST_SCORE][idx]))
        statistic['train'].append(float(cv_res.cv_results_[TRAIN_SCORE][idx]))

    if verbose:
        print(f'DataSet processed with {cv_res.cv_results_[TRAIN_SCORE][cv_res.best_index_]} train score, '
              f'and {cv_res.cv_results_[TEST_SCORE][cv_res.best_index_]} test score, depth is {params[MAX_DEPTH]}')
    return params, statistic


def dt(data_sets: List[DSWithSplit], verbose=False):
    params_and_stat = []
    for i in tqdm(range(len(data_sets))):
        params, statistic = process_data_set(data_sets[i], verbose)
        params_and_stat.append((i, params, statistic))

    def more_than(e1, e2):
        if e1 is None:
            return True
        if e2 is None:
            return True

        _, p1, s1 = e1
        _, p2, s2 = e2

        if p1[MAX_DEPTH] != p2[MAX_DEPTH]:
            return int(p1[MAX_DEPTH]) > int(p2[MAX_DEPTH])

        depth_i = DEPTH_CHOOSE.index(int(p1[MAX_DEPTH]))

        if s1['test'][depth_i] > s2['test'][depth_i]:
            return True
        return False

    min_depth_tree = None
    max_depth_tree = None

    for tree_stat in params_and_stat:
        if more_than(tree_stat, max_depth_tree):
            max_depth_tree = tree_stat
        if more_than(min_depth_tree, tree_stat):
            min_depth_tree = tree_stat

    if verbose:
        print(f"Got min stat {min_depth_tree}")
        print(f"Got max stat {max_depth_tree}")

    def get_title(e):
        idx, p, s = e
        title = f'DataSet #{idx + 1}'
        title += ". Params: "
        title += f"{CRITERION}='{p[CRITERION]}',{SPLITTER}='{p[SPLITTER]}'"
        return title

    metric_plot(min_depth_tree[-1], DEPTH_CHOOSE, title=get_title(min_depth_tree))
    metric_plot(max_depth_tree[-1], DEPTH_CHOOSE, title=get_title(max_depth_tree))


if __name__ == '__main__':
    dss = log_action('loading data sets', lambda: read_data_sets())
    dt(dss)
