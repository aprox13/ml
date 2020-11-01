import matplotlib.pyplot as plt
import pandas as pd


def hist(data: dict, index, title='', x_label='', y_label=''):
    df = pd.DataFrame(data, index=index)
    df.plot(kind='bar')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
