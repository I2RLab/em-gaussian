# -*- coding:   utf-8 -*-
# Created on Jun 20 2019 EDT
# @author: Maziar Fooladi Mahani

import numpy as np


# def A_matrix(i,j,t):
    # wb, wp, wd =
    #
    # mu =
#

def forward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]

    alpha = np.zeros((N, S))

    # base case
    for s2 in range(S):
        alpha[0, :] = pi * np.sum(O[:, s2, observations[0]])

    # recursive case
    for i in range(1, N):
        for s2 in range(S):
            for s1 in range(S):
                alpha[i, s2] += alpha[i - 1, s1] * A[s1, s2] * O[s1, s2, observations[i]]

    return alpha, np.sum(alpha[N - 1, :])


def backward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]

    beta = np.zeros((N, S))

    # base case
    beta[N - 1, :] = 1

    # recursive case
    for i in range(N - 2, -1, -1):
        for s1 in range(S):
            for s2 in range(S):
                beta[i, s1] += beta[i + 1, s2] * A[s1, s2] * O[s1, s2, observations[i + 1]]

    return beta, np.sum(pi * O[:, observations[0]] * beta[0, :])


def baum_welch(training, pi, A, O, iterations):
    pi, A, O = np.copy(pi), np.copy(A), np.copy(O)  # take copies, as we modify them
    S = pi.shape[0]

    # do several steps of EM hill climbing
    for it in range(iterations):
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(A)
        O1 = np.zeros_like(O)

        for observations in training:
            # compute forward-backward matrices
            alpha, za = forward((pi, A, O), observations)
            # print('alpha=\n', alpha)
            # print('za=\n', za)
            beta, zb = backward((pi, A, O), observations)
            # print('beta=\n', beta)
            # print('zb=\n', zb)

            assert abs(za - zb) < 1e-2, "it's badness 10000 if the marginals don't agree"

            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            pi1 += alpha[0, :] * beta[0, :] / za
            for i in range(0, len(observations)):
                O1[:, observations[i]] += alpha[i, :] * beta[i, :] / za
            for i in range(1, len(observations)):
                for s1 in range(S):
                    for s2 in range(S):
                        A1[s1, s2] += alpha[i - 1, s1] * A[s1, s2] * O[s1, s2, observations[i]] * beta[i, s2] / za

        # normalise pi_new, h_ijt, g_jlt
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            A[s, :] = A1[s, :] / np.sum(A1[s, :])
            O[s, :] = O1[s, :] / np.sum(O1[s, :])
        print('A=\n',A,'\n')
        print('O=\n',O,'\n')

    return pi, A, O


A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])
pi = np.array([0.5, 0.2, 0.3])
O = np.array([[[0.15, 0.15, 0.1], [0.05, 0.05, 0.1], [0.1, 0.2, 0.1]], [[0.05, 0.1, 0.15], [0.15, 0.05, 0.05], [0.05, 0.1, 0.3]], [[0.1, 0.1, 0.2], [0.15, 0.05, 0.05], [0.05, 0.1, 0.1]]])

states = UP, DOWN, UNCHANGED = 0, 1, 2

pi3, A3, O3 = baum_welch([[UNCHANGED, UP, DOWN], [DOWN, DOWN, UP, UNCHANGED, UNCHANGED, DOWN, UP, UP]], pi, A, O, 10)

