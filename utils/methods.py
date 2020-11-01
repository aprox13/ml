import time
from functools import reduce
from typing import TypeVar, Callable

T = TypeVar('T')

def group_by(f, lst):
    res = {}

    for e in lst:
        key = f(e)
        if key in res:
            res[key].append(e)
        else:
            res[key] = [e]
    return res


def find_in(array, by, default=None):
    return next(filter(by, array), default)


def in_range(x, start_ex, end_ex):
    return start_ex < x < end_ex


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


def log_action(action_name, action: Callable[[], T], with_start_msg=False, with_result=True) -> T:
    def millis():
        return int(round(time.time() * 1000))

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
