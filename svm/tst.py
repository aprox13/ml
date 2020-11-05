import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm


def my_func(a):
    print(f"on a {a}")
    return (a[0] + a[-1]) * 0.5


b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(b)

print("0")
np.apply_along_axis(my_func, 0, b)

print(1)
np.apply_along_axis(my_func, 1, tqdm(b))
