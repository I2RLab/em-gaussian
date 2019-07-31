import numpy as np
import matplotlib.pyplot as plt

input_seq = np.random.rand(10, 3)
input_seq_r = np.transpose(input_seq)

time_seq = np.arange(len(input_seq))
time_length = len(time_seq)

output_seq = np.random.randint(0, 8, (time_length, 1))

# print(output_seq)
# plot the input sequence
for i, u_t in enumerate(input_seq_r):
        plt.plot(time_seq, u_t)


pi = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])

state_vec = np.arange(1, 9).reshape((1, 8))

w_transition = np.ndarray((8, 8))
# w_transition = [w_mb, w_ms, w_m10, w_m20, w_m30, w_m11, w_m21, w_m31]
w_transition[0, :] = [0.010, 0.20, 0.030, -0.030, 0.040, -0.040, 0.050, -0.050]
w_transition[1, :] = [0.011, 0.21, 0.031, -0.031, 0.041, -0.041, 0.051, -0.051]
w_transition[2, :] = [0.012, 0.22, 0.032, -0.032, 0.042, -0.042, 0.052, -0.052]
w_transition[3, :] = [0.013, 0.23, 0.033, -0.033, 0.043, -0.043, 0.053, -0.053]
w_transition[4, :] = [0.014, 0.24, 0.034, -0.034, 0.044, -0.044, 0.054, -0.054]
w_transition[5, :] = [0.015, 0.25, 0.035, -0.035, 0.045, -0.045, 0.055, -0.055]
w_transition[6, :] = [0.016, 0.26, 0.036, -0.036, 0.046, -0.046, 0.056, -0.056]
w_transition[7, :] = [0.017, 0.27, 0.037, -0.037, 0.047, -0.047, 0.057, -0.057]

w_emission_int = np.ndarray((8, 2))
# w_emission_int = [w_be, w_se]
w_emission_int[0, :] = [0.010, 0.20]
w_emission_int[1, :] = [0.011, 0.21]
w_emission_int[2, :] = [0.012, 0.22]
w_emission_int[3, :] = [0.013, 0.23]
w_emission_int[4, :] = [0.014, 0.24]
w_emission_int[5, :] = [0.015, 0.25]
w_emission_int[6, :] = [0.016, 0.26]
w_emission_int[7, :] = [0.017, 0.27]


def mlogit_transition(w, u):
    time_len = len(u)
    x_matrix = np.ones((1, 8))
    y_matrix = state_vec
    z_matrix = np.concatenate((x_matrix, y_matrix))

    # print(np.ones((1,8)) * u[0, 0])
    # print('u\n', u)

    a = np.ones((8, 3))
    E_matrix = []
    for t, u_t in enumerate(u):
        try:
            b = np.multiply(a, u_t)
            c = np.multiply(a, u[t+1])

            e_matrix = np.concatenate((z_matrix, np.transpose(b), np.transpose(c)))
            # print('e_matrix=')
            # print(e_matrix, e_matrix.shape)
            E_matrix.append(np.transpose(e_matrix))
        except:
            pass

    a_ijt = np.empty((1, 8))

    for t in range(len(E_matrix)):
        a_ij = np.empty((1, 8))

        for ix, x in enumerate(E_matrix[0]):
            beta = list()
            for iw, w_m in enumerate(w):
                beta.append(np.exp(np.matmul(w_m, x)))

            den = 1 + sum(beta[0:-1])
            beta /= den
            beta[-1] = 1. / den

            a_ij = np.concatenate((a_ij, np.array(beta).reshape(1, 8)))

        a_ij = a_ij[1::]

        a_ijt = np.concatenate((a_ijt, a_ij))
        # print('aij')
        # print(a_ij)
        # print('sum')
        # print(np.sum(a_ij, 1))
        # print()

    a_ijt = a_ijt[1::]
    # a_ijt = np.reshape(a_ijt, (8, 8, 9))

    # a_ijt = a_ijt.reshape((9, 8, 8))

    # print(a_ijt, a_ijt.shape)

    return a_ijt


def mlogit_emission_int(w, o):
    x_matrix = np.ones((1, 8))
    y_matrix = state_vec
    z_matrix = np.concatenate((x_matrix, y_matrix))
    z_matrix = np.transpose(z_matrix)

    b_lj = np.empty((1, 8))

    for ix, x in enumerate(z_matrix):
        beta_e = list()
        for iw, w_m in enumerate(w):
            beta_e.append(np.exp(np.matmul(w_m, x)))

        den = 1 + sum(beta_e[0:-1])
        beta_e /= den
        beta_e[-1] = 1. / den

        b_lj = np.concatenate((b_lj, np.array(beta_e).reshape(1, 8)))

    b_lj = b_lj[1::]

    b_jt = np.empty((1, 8))

    for t in range(time_length):
        # print('1',b_jt.shape)
        print('ot', o[t])
        # print('2',b_lj[o[t]].shape)
        b_jt = np.concatenate((b_jt, b_lj[o[t]]))

    b_jt = b_jt[1::]

    return b_jt


def forward(params, observations):
    pi, A, O = params
    N = time_length
    S = pi.shape[0]

    alpha = np.zeros((N, S))
    # base case
    for s in range(S):
        alpha[0, :] = pi * O[0]

    # recursive case
    # print('A\n', A)
    for k in range(1, N):
        for s1 in range(S):
            alpha[k] += alpha[k - 1, s1] * A[k, s1] * O[k]

    return alpha, np.sum(alpha[N - 1, :])


def backward(params, observations):
    pi, A, O = params
    N = time_length
    S = pi.shape[0]

    beta = np.zeros((N, S))

    # base case
    beta[N - 1, :] = 1

    # recursive case
    for k in range(N - 2, -1, -1):
        for s1 in range(S):
            beta[k] += beta[k + 1, s1] * A[k, s1] * O[k]

    return beta, np.sum(pi * O[0] * beta[0, :])


def baum_welch(training, pi, iterations, input_seq, w_transition, w_emission_int):
    A = mlogit_transition(w_transition, input_seq)
    O = mlogit_emission_int(w_emission_int, output_seq)

    pi, A, O = np.copy(pi), np.copy(A), np.copy(O)  # take copies, as we modify them
    S = pi.shape[0]

    # do several steps of EM hill climbing
    for it in range(iterations):
        print('iteration=', it)
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(A)
        O1 = np.zeros_like(O)

        for obs_t, observations in enumerate(training):
            # compute forward-backward matrices
            alpha, za = forward((pi, A, O), observations)
            print('alpha\n', alpha)
            print('za\n', za)
            beta, zb = backward((pi, A, O), observations)
            print('beta\n', beta)
            print('zb\n', zb)

            # assert abs(za - zb) < 1e-1, "it's badness 10 if the marginals don't agree"

            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            pi1 += alpha[0, :] * beta[0, :] / za


            for i in range(0, obs_t):
                O1[i] += alpha[i, :] * beta[i, :] / za

            for i in range(1, obs_t):
                for s1 in range(S):
                    A1[s1] += alpha[i - 1, s1] * A[i, s1] * O[i] * beta[i] / za

            # print('A1\n', A1)

        # normalise pi1, A1, O1
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            A[s, :] = A1[s, :] / np.sum(A1[s, :])
            O[s, :] = O1[s, :] / np.sum(O1[s, :])
        print('A=\n', A, '\n')
        print('O=\n', O, '\n')

    return pi, A, O


baum_welch(output_seq, pi, 10, input_seq, w_transition, w_emission_int)

plt.show()