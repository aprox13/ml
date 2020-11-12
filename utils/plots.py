from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle
import statistics

COLORS = [
    "#BA68C8",
    "#e57373",
    "#F06292",
    "#64B5F6",
    "#4DB6AC",
    "#FFB74D",
    "#90A4AE"
]

DARK = [
    "#311B92",
    "#263238",
    "#004D40"
]


def colors(cs):
    i = 0
    lst = [c for c in cs]
    shuffle(lst)
    while True:
        i += 1
        yield lst[i % len(lst)]


def hist(data: dict, index, title='', x_label='', y_label=''):
    """
    Построение гистограммы

    :param data: словарь { key -> [value] } - где длина занчений равна длине индексов
    :param index: индексы по оси X (значения)
    :param title: название графика
    :param x_label: подпись по X
    :param y_label: подпись по Y
    """
    df = pd.DataFrame(data, index=index)
    df.plot(kind='bar')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def metric_plot(data: dict, x_values: List, title='', x_label='', metric='Accuracy', y_extend=0.2, with_text=True,
                default_color=False, fit_x=False, n_col=2):
    y_max = 0
    y_min = 2
    for v in data.values():
        y_max = max(y_max, max(v))
        y_min = min(y_min, min(v))

    dy = y_max - y_min

    y_max += dy * y_extend
    y_min = max(0, y_min - dy * y_extend)

    for_data = colors(COLORS)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.ylim(y_min, y_max)

    text_shift = dy * 0.025

    maximums = []
    for k, v in data.items():
        max_value = max(v)
        target_x = x_values[v.index(max_value)]

        maximums.append((target_x, max_value, k))
        xv = list(range(len(x_values))) if fit_x else x_values
        if fit_x:
            plt.xticks(xv, labels=x_values)
        if default_color:
            plt.plot(xv, v)
        else:
            plt.plot(xv, v, color=next(for_data))

    if not fit_x:
        xx = []
        yy = []
        x_med = statistics.median(x_values)
        for x, y, k in maximums:
            xx.append(x)
            yy.append(y)
            txt = f"{k}, depth: {x}\n{metric}: {y}"
            ha = 'left' if x < x_med else 'right'
            if with_text:
                plt.text(x, y + text_shift, txt,
                         horizontalalignment=ha,
                         verticalalignment='bottom')

        plt.scatter(xx, yy, marker='x', color='#606060')

    plt.legend(list(data.keys()), loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=n_col)
    plt.show()
