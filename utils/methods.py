import time
from functools import reduce
from random import shuffle
from typing import TypeVar, Callable
import numpy as np

T = TypeVar('T')


def pretty_time(millis: int) -> str:
    base = [(1000 * 60, "min"), (1000, "sec"), (1, "ms")]

    def step(acc, x):
        cur_millis, result = acc
        multiplier, name = x

        part = cur_millis // multiplier
        if part != 0:
            result.append(f"{part}{name}")
            cur_millis -= part * multiplier
            return cur_millis, result
        return acc

    res = reduce(step, base, (millis, []))[1]
    if len(res) != 0:
        return "".join(res)
    return "0ms"


def log_action(action_name, action: Callable[[], T], with_start_msg=False, with_result=True, verbose=True) -> T:
    def millis():
        return int(round(time.time() * 1000))

    if not verbose:
        return action()

    if with_start_msg:
        print(f"starting '{action_name}'")

    start = millis()
    res = action()
    end_time_s = pretty_time(millis() - start)
    result_part = ""
    if with_result:
        result_part = f" with result {res}"

    print(f"'{action_name}' ends in {end_time_s}{result_part}")
    return res


def has_all(**kwargs):
    return dict_contains(kwargs)


def dict_contains(d: dict):
    def contains_inner(outer_dict: dict):
        for k, v in d.items():
            if k not in outer_dict or outer_dict[k] != v:
                return False
        return True

    return contains_inner


def index_where(predicate, lst):
    r = indices_where(predicate, lst)
    return -1 if len(r) == 0 else r[0]


def indices_where(predicate, lst):
    res = []
    for i in range(len(lst)):
        if predicate(lst[i]):
            res.append(i)
    return res


def filter_key(f, d: dict):
    res = {}
    for k, v in d.items():
        if f(k):
            res[k] = v
    return res


def join(d1: dict, d2: dict) -> dict:
    res = {}

    def join_inner(d):
        for k, v in d.items():
            res[k] = v

    join_inner(d1)
    join_inner(d2)
    return res
