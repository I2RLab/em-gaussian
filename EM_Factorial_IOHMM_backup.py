
import numpy as np
import xlrd

np.set_printoptions(linewidth=600)
np.set_printoptions(precision=2, edgeitems=25)


workbook = xlrd.open_workbook('IO_sample11.xlsx')
worksheet = workbook.sheet_by_index(0)

state_scale = 2

agent_num = 3

input_num = 3

output_num = 8

state_total = state_scale ** agent_num

input_tot = input_num ** agent_num

pi = np.ones((state_total,)) / state_total  # initial distribution

data_sample_i = list()

state_vec = np.arange(1, state_total + 1).reshape((1, state_total))

for i in range(agent_num):
    data_sample_i.append(worksheet.col_values(i))

data_input = np.transpose(np.array(data_sample_i))

data_output = np.transpose(np.array(worksheet.col_values(4))).reshape((len(worksheet.col_values(4)),1))

# sampled inputs
input_seq = data_input

time_seq = np.arange(len(input_seq))  # time sequence
time_length = len(time_seq)

# random output seq
output_seq = data_output
output_lambda = dict()  # output_lambda(t) = 1 when the t'th output is l

for i in range(1, output_num + 1):
    output_lambda[i] = np.where(output_seq == i)[0]


def sigma_input(input_seq, t_len):
    input_k = dict()

    for i1 in range(1, input_num + 1):
        for i2 in range(1, input_num + 1):
            for i3 in range(1, input_num + 1):
                input_k[i1, i2, i3] = []
                
                for k in range(t_len):
                    if list(input_seq[k]) == [i1, i2, i3]:
                        input_k[i1, i2, i3].append(1)
                
                    else:
                        input_k[i1, i2, i3].append(0)

    return input_k


sigma_k = sigma_input(input_seq, time_length)
input_lambda = dict()

array = np.array(list(sigma_k.values()))

for u, t in enumerate(array):
    input_lambda[u] = list(np.where(t == 1)[0])

io_lambda = dict()

for i, ti in enumerate(input_lambda):
    for o, to in enumerate(output_lambda):
        io_lambda[ti, to] = list(set(input_lambda[ti]).intersection(output_lambda[to]))

print('io_lambda')
print(io_lambda)
#####################################################################
# plot input sequence
# plt.subplots(1, 1, sharex='all', sharey='all')
# plt.subplot(211)
#
# for i, u_t in enumerate(np.transpose(input_seq)):
#     plt.plot(time_seq, u_t)
#     plt.grid(color='b', axis='y')

# plt.subplot(212)
# plt.plot(time_seq, output_seq, '.')

# plt.show()
#####################################################################

# w_transition = [w_b, w_s, w_x1, w_x2, w_x3]
w_transition = np.ndarray((8, 5))
w_transition[0, :] = [-1.5, -.1, .3, .3, .3]
w_transition[1, :] = [-.5, -.1, -.3, .3, .3]
w_transition[2, :] = [-.5, -.1, .3, -.3, .3]
w_transition[3, :] = [-.5, -.1, .3, .3, -.3]
w_transition[4, :] = [0.0, .1, -.3, -.3, .3]
w_transition[5, :] = [0.0, .1, -.3, .3, -.3]
w_transition[6, :] = [0.0, .1, .3, -.3, -.3]
w_transition[7, :] = [0.0, .1, -.3, -.3, -.3]

w_observation = np.ndarray((8, 5))
w_observation[0, :] = [-.7, -.1, .1, .1, .1]
w_observation[1, :] = [0.0, -.1, -.1, .1, .1]
w_observation[2, :] = [0.0, -.1, .1, -.1, .1]
w_observation[3, :] = [0.0, -.1, .1, .1, -.1]
w_observation[4, :] = [-.2, 0.1, -.1, -.1, .1]
w_observation[5, :] = [-.2, 0.1, -.1, .1, -.1]
w_observation[6, :] = [-.2, 0.1, .1, -.1, -.1]
w_observation[7, :] = [-.2, 0.1, -.1, -.1, -.1]


def mlogit_transition(w, u):
    x_matrix = np.ones((1, state_total))
    y_matrix = state_vec
    z_matrix = np.concatenate((x_matrix, y_matrix))
    a = np.ones((state_total, agent_num))
    E_matrix = []

    for t, u_t in enumerate(u):
        try:
            c = np.multiply(a, u_t)
            e_matrix = np.concatenate((z_matrix, np.transpose(c)))
            E_matrix.append(np.transpose(e_matrix))

        except NameError:
            pass

    a_ijt = np.ones((state_total, state_total)) / state_total

    for t in range(len(E_matrix)):
        a_ij = np.empty((1, state_total))

        for ix, x in enumerate(E_matrix[t]):
            beta = list()
            for iw, w_m in enumerate(w):
                beta.append(np.exp(np.matmul(w_m, x)))

            den = 1 + sum(beta[0:-1])
            beta /= den
            beta[-1] = 1. / den

            a_ij = np.concatenate((a_ij, np.array(beta).reshape(1, state_total)))

        a_ij = a_ij[1::]

        a_ijt = np.concatenate((a_ijt, a_ij))

    A_ijt = a_ijt.reshape((int(len(a_ijt) / state_total), state_total, state_total))
    A_ijt = A_ijt[1::]

    return A_ijt


def mlogit_observation(w, output, input):
    z_matrix = np.concatenate((np.ones((1, state_total)), state_vec))
    a = np.ones((state_total, 3))
    E_matrix = []

    for t, u_t in enumerate(input):
        try:
            b = np.multiply(a, input[t])
            e_matrix = np.concatenate((z_matrix, np.transpose(b)))
            E_matrix.append(np.transpose(e_matrix))

        except:
            pass

    b_jt = np.ones((1, state_total))

    for t in range(len(E_matrix)):
        b_ij = np.empty((1, state_total))

        for iw, w_m in enumerate(w):
            beta = list()
            for ix, x in enumerate(E_matrix[t]):
                beta.append(np.exp(np.matmul(w_m, x)))  # w_l = [w_lb, w_ls, w_l1, w_l2, w_l3] & x = [1, S(t), u1, u2, u3]

            den = 1 + sum(beta[0:-1])
            beta /= den
            beta[-1] = 1. / den

            b_ij = np.concatenate((b_ij, np.array(beta).reshape(1, state_total)))
        b_ij = b_ij[1::]
        b_ij = np.transpose(b_ij)
        b_jt = np.concatenate((b_jt, b_ij[[int(output[t][0])-1]]))

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
        for j in range(S):
            for i in range(S):
                if np.isnan(alpha[k - 1, i]) or np.isnan(A[k, i, j]) or np.isnan(O[k,j]):
                    pass
                
                else:
                    alpha[k, j] += alpha[k - 1, i] * A[k, i, j] * O[k, j]

    return alpha, max(np.sum(alpha[N - 1, :]), 10 ** -300)


def backward(params):
    pi, A, O = params
    N = time_length
    S = pi.shape[0]

    beta = np.zeros((N, S))

    # base case
    beta[N - 1, :] = 1

    # recursive case
    for k in range(N - 2, -1, -1):
        for i in range(S):
            for j in range(S):
                beta[k, i] += beta[k + 1, j] * A[k + 1, i, j] * O[k + 1, j]

    return beta, max(np.sum(pi * O[0] * beta[0, :]), 10 ** -300)


def baum_welch(output_seq, pi, iterations, input_seq, w_transition, w_obs):
    A = mlogit_transition(w_transition, input_seq)
    O = mlogit_observation(w_obs, output_seq, input_seq)
    print('A init\n', A)
    print('O init\n', O)
    
    pi, A, O = np.copy(pi), np.copy(A), np.copy(O)  # take copies, as we modify them
    S = pi.shape[0]
    obs_length = int(len(A))
    
    # do several steps of EM hill climbing
    for it in range(iterations):
        print('iteration=', it)
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(A)
        A1_New = np.zeros_like(A)
        O1 = np.zeros((obs_length, S))
    
        # compute forward-backward matrices
        alpha, za = forward((pi, A, O))
        beta, zb = backward((pi, A, O))
        # print('alpha\n', alpha)
        print('za\n', za)
        # print('beta\n', beta)
        print('zb\n', zb)
    
        assert abs(za - zb) < 1e-2, "it's badness 10000 if the marginals don't agree"
    
        # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
        pi1 += alpha[0, :] * beta[0, :] / za
        pi = pi1 / np.sum(pi1)  # normalise pi1

        for k in range(0, obs_length):
            O1[k] += alpha[k, :] * beta[k, :] / za
    
        for k in range(1, obs_length):
            for j in range(S):
                for i in range(S):
                    A1[k - 1, i, j] = alpha[k - 1, i] * A[k, i, j] * O[k, j] * beta[k, j] / za

        for k, ti in enumerate(input_lambda.values()):
            H = np.zeros_like(A[0]) + 10 ** -300
            if len(ti) > 0:
                for ki, t in enumerate(ti):
                    H += A1[t]
            
                H_temp = np.transpose(np.transpose(H) / np.sum(H, 1))
            
                for ki, t in enumerate(ti):
                    A1_New[t] = H_temp
    
        O_New = np.zeros_like(O)
        w_jl = np.zeros((output_num, state_total)) + 10 ** -250
        
        for t1, to in enumerate(output_lambda):
            if len(output_lambda[to]) > 0:
                O1_temp = np.zeros((1, state_total))
                for t, ut in enumerate(output_lambda[to]):
                    O1_temp += O1[ut]
        
                w_jl[t1] = O1_temp
            else:
                w_jl[t1] = np.zeros_like(O[0]) + 10 ** -250
                
        w_jl /= np.sum(w_jl, 0)
        
        for t1, to in enumerate(output_lambda):
            if len(output_lambda[to]) > 0:
                for t, ut in enumerate(output_lambda[to]):
                    O_New[ut] = w_jl[t1]
    
        # uncomment to compute O_jlk instead
        # w_ilk = np.zeros((input_tot, output_num, state_total)) + 10 ** -300
    
        # for k, ti in enumerate(input_lambda):
        #     for l, to in enumerate(output_lambda):
        #         if len(io_lambda[ti, to]) > 0:
        #             w_ilk_temp = np.zeros((1, state_total))
        #             for i, ts in enumerate(io_lambda[ti, to]):
        #                 w_ilk_temp += O1[ts]
        #             w_ilk[ti, to - 1, :] = w_ilk_temp
        #
        # for k, ti in enumerate(input_lambda):
        #     if np.sum(w_ilk[ti]) > 0:
        #         w_ilk[ti] /= np.sum(w_ilk[ti], 0)
        #         for l, to in enumerate(output_lambda):
        #             if len(io_lambda[ti, to]) > 0:
        #                 for kl, ts in enumerate(io_lambda[ti, to]):
        #                     O_New[ts] = w_ilk[ti, to-1]
    
        A, O = A1_New, O_New
        # print('A=\n', A, '\n')
        # print('O=\n', O, '\n')
    
    A_ijk = dict()

    for k, ti in enumerate(input_lambda.values()):
        if len(ti) > 0:
            A_ijk[k] = A[ti[0]]
        else:
            A_ijk[k] = np.zeros_like(A[0])

    O_jl = dict()

    for l, to in enumerate(output_lambda):
        if len(output_lambda[to]) > 0:
            O_jl[l] = O[output_lambda[to][0]]
        else:
            O_jl[l] = np.zeros_like(O[0])
    
    # O_jlk = dict()
    # for k, ti in enumerate(input_lambda):
    #     for l, to in enumerate(output_lambda):
    #         if len(io_lambda[ti, to]) > 0:
    #             O_jlk[k, l] = O[io_lambda[ti, to][0]]
    #         else:
    #             O_jlk[k, l] = np.zeros_like(O[0])

    print('A=\n', A, '\n')
    print('O=\n', O, '\n')
    return pi, A, O, A_ijk, O_jl


if __name__ == "__main__":
    
    pi_trained, A_trained, O_trained, A_ijk, O_jl = baum_welch(output_seq, pi, 7, input_seq, w_transition, w_observation)
    # plt.plot(np.transpose(A_trained[-1]),'*')
    # plt.show()
