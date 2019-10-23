import numpy as np

x1 = 4
x2 = 4
x3 = 5

w12 = np.ones((x1, x2))
w13 = np.ones((x1, x3))


def net_input_ij(i, j):
    for u in range(x1):
        if u != i:
            for v in range(x2):
                