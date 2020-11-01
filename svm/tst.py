a = [1, 5]


def in_range(x, arr):
    return min(arr) <= x <= max(arr)


def bound(x, mn, mx):
    return min(
        mx,
        max(x, mn)
    )

t = [-2, 10]

for i in range(t[0], t[1] + 1):
    b = bound(i, min(a), max(a))
    print(f"for x = {i}, bound = {b}, bound in range is {in_range(b, a)}")