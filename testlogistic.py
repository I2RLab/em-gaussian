import numpy as np

weights = np.arange(64).reshape((8, 8))/100.

state = np.arange(1,9)
input = np.array([3., 4., 5., 6., 7., 8.])

def mlogit():
    a_mat =[]
    x = np.ones((1, 8))
    x[0, 2::] = input
    for j, s1 in enumerate(state):
        beta = []
        for i, s0 in enumerate(state):
            x[0, 1] = s0

            wx = np.matmul(x[0], weights[j])
            beta.append(np.exp(wx))
        beta[-1] = 1. / (1. + sum(beta[0:-1]))
        beta /= (1. + sum(beta[0:-1]))

        a_mat.append(beta)


mlogit()
