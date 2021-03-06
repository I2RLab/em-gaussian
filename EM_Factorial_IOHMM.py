import numpy as np
import matplotlib.pyplot as plt
import collections as col
import xlsxwriter
import xlrd


workbook = xlrd.open_workbook('IO_sample.xlsx')
worksheet = workbook.sheet_by_index(0)
data_sample_i = list()

for i in range(3):
    data_sample_i.append(worksheet.col_values(i))

data_input = np.transpose(np.array(data_sample_i))
# data_input = data_input[0:4]

# temp random robots' performances {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
# input_seq = np.random.randint(0, 6, (300, 3)) / 5.

# sampled inputs
input_seq = data_input

input_seq_r = np.transpose(input_seq)  # input sequence transpose

time_seq = np.arange(len(input_seq))  # time sequence
time_length = len(time_seq)

# random output seq
output_seq = np.random.randint(0, 8, (time_length, 1))


# sampled output
# output_seq = np.transpose(np.array(worksheet.col_values(4))).reshape(300, 1)


output_lambda = dict()  # output_lambda(t) = 1 when the t'th output is l

for i in range(343):
    output_lambda[i] = np.where(output_seq == i)[0]

# print('output lambda', output_lambda)


def sigma_input(input_seq, t_len):

    input_k = dict()

    for i1 in range(6):
        for i2 in range(6):
            for i3 in range(6):
                input_k[i1 / 5, i2 / 5, i3 / 5] = []
                for k in range(t_len):
                    if list(input_seq[k]) == [i1 / 5, i2 / 5, i3 / 5]:
                        input_k[i1 / 5., i2 / 5., i3 / 5.].append(1)
                    else:
                        input_k[i1 / 5., i2 / 5., i3 / 5.].append(0)

    # uncomment to write the input sequence onto a spreadsheet
    # workbook = xlsxwriter.Workbook('Input.xlsx')
    # worksheet = workbook.add_worksheet()

    # for i, inp in enumerate(input_k):
    #     worksheet.write_row(i, 0, list(input_k.values())[i])
    # workbook.close()

    return input_k


sigma_k = sigma_input(input_seq, time_length)
sigma_u = dict()  # input_lambda is a dict to show what time instances each input was observed
array = np.array(list(sigma_k.values()))

for u, t in enumerate(array):
    sigma_u[u] = list(np.where(t == 1)[0])

plt.subplots(1, 1, sharex='all', sharey='all')

# plot the input sequence
for i, u_t in enumerate(input_seq_r):
    # plt.subplot(int('31{}'.format(i+1)))

    plt.plot(time_seq, u_t)
    # plt.grid(color='b', axis='y')


plt.show()

# pi = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])  # initial distribution

# w_transition = [w_mb, w_ms, w_m10, w_m20, w_m30, w_m11, w_m21, w_m31]
# w_transition = np.ndarray((8, 8))
# w_transition[0, :] = [0.01, 0.04, 0.02, 0.02, 0.02, -0.02, -0.02, -0.02]
# w_transition[1, :] = [0.01, 0.03, 0.10, 0.02, 0.02, -0.08, -0.02, -0.02]
# w_transition[2, :] = [0.01, 0.03, 0.02, 0.10, 0.02, -0.02, -0.08, -0.02]
# w_transition[3, :] = [0.01, 0.03, 0.02, 0.02, 0.10, -0.02, -0.02, -0.08]
# w_transition[4, :] = [0.01, 0.01, 0.05, 0.05, 0.02, -0.04, -0.04, -0.02]
# w_transition[5, :] = [0.01, 0.01, 0.05, 0.02, 0.05, -0.04, -0.02, -0.04]
# w_transition[6, :] = [0.01, 0.01, 0.02, 0.05, 0.05, -0.02, -0.04, -0.04]
# w_transition[7, :] = [-0.01, -0.1, 0.02, 0.02, 0.02, -0.02, -0.02, -0.02]

# w_observation = np.ndarray((8, 2))
# w_observation = [w_be, w_se]
# w_observation[0, :] = [0.21, 0.31]
# w_observation[1, :] = [0.08, 0.21]
# w_observation[2, :] = [0.08, 0.21]
# w_observation[3, :] = [0.08, 0.21]
# w_observation[4, :] = [0.04, 0.11]
# w_observation[5, :] = [0.04, 0.11]
# w_observation[6, :] = [0.04, 0.11]
# w_observation[7, :] = [-0.10, 0.31]

state_scale = 7
agent_num = 3

state_total = state_scale ** agent_num

pi = np.ones((state_total,)) / state_total  # initial distribution

w_transition = np.ndarray((state_total, 5))  # w_transition = [w_mb, w_ms, w_x1, w_x2, w_x3]

for i in range(state_total):
    w_transition[i, :] = [-6., 1.5, 1., 1., 1.]

w_observation = np.ndarray((state_total, 2))  # w_observation = [w_b, w_s]

state_vec = np.arange(1, state_total + 1).reshape((1, state_total))

for i in range(state_total):
    w_observation[i, :] = [-8, 2]


def mlogit_transition(w, u):
    x_matrix = np.ones((1, state_total))
    y_matrix = state_vec
    z_matrix = np.concatenate((x_matrix, y_matrix))
    a = np.ones((state_total, 3))
    E_matrix = []

    for t, u_t in enumerate(u):
        try:
            # b = np.multiply(a, u_t)
            c = np.multiply(a, u[t + 1])
            e_matrix = np.concatenate((z_matrix, np.transpose(c)))
            E_matrix.append(np.transpose(e_matrix))

        except:
            pass

    a_ijt = np.ones((state_total, state_total)) / state_total

    for t in range(len(E_matrix)):
        print('t', t)
        a_ij = np.empty((1, state_total))

        for ix, x in enumerate(E_matrix[0]):
            beta = list()
            for iw, w_m in enumerate(w):
                beta.append(np.exp(np.matmul(w_m, x)))  # w_m = [w_mb, w_ms, w_x1, w_x2, w_x3] & x = [1, S(t-1), x1, x2, x3]

            den = 1 + sum(beta[0:-1])
            beta /= den
            beta[-1] = 1. / den

            a_ij = np.concatenate((a_ij, np.array(beta).reshape(1, state_total)))

        a_ij = a_ij[1::]

        a_ijt = np.concatenate((a_ijt, a_ij))

    A_ijt = a_ijt.reshape((int(len(a_ijt) / state_total), state_total, state_total))

    return A_ijt


def mlogit_emission_int(w, o):
    x_matrix = np.ones((1, state_total))
    y_matrix = state_vec
    z_matrix = np.concatenate((x_matrix, y_matrix))
    z_matrix = np.transpose(z_matrix)

    b_lj = np.empty((1, state_total))

    for ix, x in enumerate(z_matrix):
        beta_e = list()
        for iw, w_m in enumerate(w):
            beta_e.append(np.exp(np.matmul(w_m, x)))

        den = 1 + sum(beta_e[0:-1])
        beta_e /= den
        beta_e[-1] = 1. / den

        b_lj = np.concatenate((b_lj, np.array(beta_e).reshape(1, state_total)))

    b_lj = b_lj[1::]

    b_jt = np.empty((1, state_total))

    for t in range(time_length):
        oo = [int(o[t][0])]
        b_jt = np.concatenate((b_jt, b_lj[oo]))

    b_jt = b_jt[1::]

    return b_jt


def forward(params):
    pi, A, O = params
    N = time_length
    S = pi.shape[0]

    alpha = np.zeros((N, S))
    # base case
    for s in range(S):
        alpha[0, :] = pi * O[0]

    # recursive case
    for k in range(1, N):
        for s1 in range(S):
            for j in range(S):
                alpha[k, s1] += alpha[k - 1, j] * A[k, j, s1] * O[k, s1]

    return alpha, np.sum(alpha[N - 1, :])


def backward(params):
    pi, A, O = params
    N = time_length
    S = pi.shape[0]

    beta = np.zeros((N, S))

    # base case
    beta[N - 1, :] = 1

    # recursive case
    for k in range(N - 2, -1, -1):
        for s1 in range(S):
            for j in range(S):
                beta[k, s1] += beta[k + 1, j] * A[k + 1, s1, j] * O[k + 1, s1]

    return beta, np.sum(pi * O[0] * beta[0, :])


def baum_welch(output_seq, pi, iterations, input_seq, w_transition, w_emission_int):
    A = mlogit_transition(w_transition, input_seq)
    O = mlogit_emission_int(w_emission_int, output_seq)
    print('A init\n', A)
    print('O init\n', O)

    pi, A, O = np.copy(pi), np.copy(A), np.copy(O)  # take copies, as we modify them
    S = pi.shape[0]
    obs_length = int(len(A) / S)

    # do several steps of EM hill climbing
    for it in range(iterations):
        print('iteration=', it)
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(A)
        H = np.zeros_like(A[0])
        H1 = np.zeros_like(A)
        O1 = np.zeros((len(O), S))

        # for i in range(N):
        # compute forward-backward matrices
        alpha, za = forward((pi, A, O))
        beta, zb = backward((pi, A, O))
        print('alpha\n', alpha)
        print('za\n', za)
        print('beta\n', beta)
        # print('zb\n', zb)

        assert abs(za - zb) < 1e-2, "it's badness 10000 if the marginals don't agree"

        # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
        pi1 += alpha[0, :] * beta[0, :] / za

        for k in range(0, obs_length):
            O1[k] += alpha[k, :] * beta[k, :] / za

        for k in range(1, obs_length):
            for i in range(S):
                for j in range(S):
                    A1[k - 1, j, i] = alpha[k - 1, j] * A[k, j, i] * O[k, i] * beta[k, i] / za

        # normalise pi_new
        pi = pi1 / np.sum(pi1)

        for k, u in enumerate(sigma_u.values()):
            if len(u) > 0:
                for i, t in enumerate(u):
                    H += A1[t]
                    if np.sum(H) > 0:
                        H1[t] = np.transpose(np.transpose(H)/np.sum(H, 1))
                    else:
                        H1[t] = 0

        OM_dict = np.zeros((state_total, state_total))
        OM1 = np.zeros_like(O)

        for i in range(state_total):
            OM = np.zeros((1, state_total))
            if len(output_lambda[i]) > 0:
                for t, l in enumerate(output_lambda[i]):
                    OM += O1[l]

                OM_dict[i] = OM

            else:
                OM_dict[i] = 0

        OM_dict /= np.sum(OM_dict, 0)

        for i in range(8):
            if len(output_lambda[i]) > 0:
                for t, l in enumerate(output_lambda[i]):
                    OM1[l] = OM_dict[i]

        A, O = H1, OM1

        print('A=\n', A, '\n')
        print('O=\n', O, '\n')
    print()
    return pi, A, O


baum_welch(output_seq, pi, 10, input_seq, w_transition, w_observation)

