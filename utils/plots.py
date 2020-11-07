from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle

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


def metric_plot(data: dict, x_values: List, title='', x_label='', metric='Accuracy', metric_max=1.3):
    for_data = colors(COLORS)
    dark = colors(DARK)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.ylim(0, metric_max)
    legend = []
    for k, v in data.items():
        max_value = max(v)
        target_x = x_values[v.index(max_value)]
        lines_c = next(dark)

        plt.axhline(y=max_value, linestyle=':', color=lines_c)
        plt.axvline(x=target_x, linestyle=':', color=lines_c)
        plt.plot(x_values, v, color=next(for_data))

        legend.append(f"{k}: {max_value}")
        legend.append(f"depth {target_x}")
        legend.append(k)
    plt.legend(legend, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.show()
