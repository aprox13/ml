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
