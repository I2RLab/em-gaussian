# -*- coding:   utf-8 -*-
# Created on Jun 20 2019 EDT
# @author: Maziar Fooladi Mahani

import numpy as np


def forward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]

    alpha = np.zeros((N, S))

    # base case
    for s in range(S):
        alpha[0, :] = pi * O[0, observations[0]]

    # recursive case
    print('A\n', A)
    for k in range(1, N):
        for s1 in range(S):
            alpha[k] += alpha[k - 1, s1] * A[k, s1] * O[k, s1, observations[k]]

    return alpha, np.sum(alpha[N - 1, :])


def backward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]

    beta = np.zeros((N, S))

    # base case
    beta[N - 1, :] = 1

    # recursive case
    for k in range(N - 2, -1, -1):
        for s1 in range(S):
            beta[k] += beta[k + 1, s1] * A[k, s1] * O[k, s1, observations[k + 1]]

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

            assert abs(za - zb) < 1e-2, "it's badness 100 if the marginals don't agree"

            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            pi1 += alpha[0, :] * beta[0, :] / za
            for i in range(0, len(observations)):
                O1[:, observations[i]] += alpha[i, :] * beta[i, :] / za
            for i in range(1, len(observations)):
                for s1 in range(S):
                    A1[s1] += alpha[i - 1, s1] * A[i, s1] * O[s1, observations[i]] * beta[i] / za

        # normalise pi_new, h_ijt, O1
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            A[s, :] = A1[s, :] / np.sum(A1[s, :])
            O[s, :] = O1[s, :] / np.sum(O1[s, :])
        print('A=\n', A, '\n')
        print('O=\n', O, '\n')

    return pi, A, O


A = np.array([[0.575, 0.025, 0.05, 0.05, 0.05, 0.025, 0.05, 0.175], [0.05, 0.625, 0.025, 0.05, 0.05, 0.05, 0.1, 0.05], [0.05, 0.075, 0.65, 0.05, 0.05, 0.025, 0.05, 0.05], [0.05, 0.05, 0.05, 0.35, 0.15, 0.1, 0.15, 0.1], [0.05, 0.05, 0.1, 0.1, 0.15, 0.35, 0.1, 0.1], [0.1, 0.15, 0.15, 0.1, 0.2, 0.1, 0.1, 0.1], [0.15, 0.2, 0.125, 0.125, 0.05, 0.15, 0.1, 0.1], [0.05, 0.05, 0.15, 0.15, 0.15, 0.1, 0.2, 0.15]])
pi = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
O = np.array([[0.05, 0.05, 0.2, 0.2, 0.15, 0.1, 0.1, 0.15], [0.15, 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.15], [0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.15, 0.1], [0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.2], [0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.25, 0.2], [0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2], [0.15, 0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1], [0.05, 0.05, 0.15, 0.1, 0.2, 0.1, 0.15, 0.2]])

input_seq = [[10,10,10,1,0,2,3,3,3,0,0,0,5,4,5,4,3,4,3,7,6,2,5,3,4,0,0,0],[10,10,10,1,0,2,3,3,3,0,0,0,5,4,5,4,3,4,3,7,6,2,5,3,4,0,0,0],[10,10,10,1,0,2,3,3,3,0,0,0,5,4,5,4,3,4,3,7,6,2,5,3,4,0,0,0]]
output_seq = [0,0,0,1,0,2,3,3,3,0,0,0,5,4,5,4,3,4,3,7,6,2,5,3,4,0,0,0]
transition_weights = np.array([[0.5, -.4, -.4, -.4], [.001, 1., -.8, -.8], [.001, -.8, 1., -.8], [.001, -.8, -.8, 1.], [.001, .55, .55, -1.7], [.001, .55, -1.7, .55], [.001, -1.7, .55, .55]])
emission_weigths = np.array([[0.5, -.4, -.4, -.4], [.001, 1., -.8, -.8], [.001, -.8, 1., -.8], [.001, -.8, -.8, 1.], [.001, .55, .55, -1.7], [.001, .55, -1.7, .55], [.001, -1.7, .55, .55]])


def transition_matrix(input_seq, weights):
    x = np.ones(4)
    aaa = []

    for t, input_agent in enumerate(input_seq):
        for input_i, input_value in enumerate(input_agent):
            x[2] = input_value
            if t == 0:
                x[3] = input_value
            else:
                x[3] = input_seq[t-1]
            aa = []
            for i in range(8):
                a = []
                den = 1
                for j in range(8):
                    x[1] = j+1
                    for k in range(7):
                        den += np.exp(np.dot(weights[k], x))

                    if j != 7:
                        p = np.exp(np.dot(weights[j], x)) / den
                    else:
                        p = 1. / den

                    a.append(p)
                aa.append(a)
            aaa.append(aa)

    return aaa


def emission_matrix(output_seq, weights):
    x = np.ones(4)
    emi = []

    for t in range(len(input_seq)):
        x[2] = input_seq[t]
        if t == 0:
            x[3] = 10
        else:
            x[3] = input_seq[t - 1]
        aa = []
        for i in range(8):
            a = []
            den = 1
            for j in range(8):
                x[1] = j + 1
                for k in range(7):
                    den += np.exp(np.dot(weights[k], x))

                if j != 7:
                    p = np.exp(np.dot(weights[j], x)) / den
                else:

                    p = 1. / den

                a.append(p)
            aa.append(a)
        emi.append(aa)

    return emi


a_ijk = transition_matrix(input_seq, transition_weights)
b_ilk = emission_matrix(output_seq, emission_weigths)
pi3, A3, O3 = baum_welch([output_seq], pi, a_ijk, b_ilk, 10)