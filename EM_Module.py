import numpy as np
import sys
import CRBM
import Training_Dataset_Generator as trainingData
import pickle
import time
from mayavi.mlab import *

sys.path.insert(0, '../Categorical_Boltzmann_Machines')

np.set_printoptions(linewidth=700)
np.set_printoptions(precision=4, edgeitems=80)
print('start', time.clock())


class EM:
    def __init__(self, iterations, input_seq, output_seq, output_f_seq, a_matrix, o_matrix, o_f_matrix, timestamp_feedback, output_f_index):
        self.a_matrix = a_matrix
        self.o_matrix = o_matrix
        self.o_f_matrix = o_f_matrix

        self.iterations = iterations

        self.state_scale = 5

        self.agent_num = 3

        self.input_num = 10

        self.output_num = 4

        self.output_f_num = 125

        self.output_f_index = output_f_index

        self.state_total = self.state_scale ** self.agent_num

        self.input_tot = self.input_num ** self.agent_num

        self.state_vec = np.arange(1, self.state_total + 1).reshape((1, self.state_total))

        self.pi = np.ones((self.state_total,)) / self.state_total  # initial distribution

        self.input_lambda = dict()

        self.output_lambda = dict()

        self.output_f_lambda = dict()

        self.N = len(input_seq)
        print('N input seq = {}'.format(self.N))

        self.N_f = len(timestamp_feedback)

        self.S = self.pi.shape[0]

        self.timestamp_feedback = timestamp_feedback
        print('timestamp feedback', self.timestamp_feedback)
        print('timestamp feedback len', len(self.timestamp_feedback))

        for i in range(1, self.output_num + 1):
            self.output_lambda[i] = np.where(output_seq == i)[0]

        # print('output f seq {}'.format(output_f_seq))
        self.feedback_length = len(output_f_seq)
        print('feedback length', self.feedback_length)

        for i in range(1, self.output_f_num + 1):
            self.output_f_lambda[i] = np.where(output_f_seq == i)[0]
            # if len(np.where(output_f_seq == i)[0]) > 0:
            #     self.feedback_length += 1

        # print('output_f lambda: {}'.format(self.output_f_lambda))

        self.input_k = dict()

        for i1 in range(1, self.input_num + 1):
            for i2 in range(1, self.input_num + 1):
                for i3 in range(1, self.input_num + 1):
                    self.input_k[i1, i2, i3] = []

                    for k in range(self.N):
                        if list(input_seq[k]) == [i1, i2, i3]:
                            self.input_k[i1, i2, i3].append(1)

                        else:
                            self.input_k[i1, i2, i3].append(0)

        self.array = np.array(list(self.input_k.values()))

        for u, t in enumerate(self.array):
            self.input_lambda[u] = list(np.where(t == 1)[0])

        self.A_init = self.transition_probability_sequence(a_matrix)
        self.O_init = self.emission_probability_sequence(o_matrix)
        # print('o f matrix: {}'.format(o_f_matrix))
        self.O_f_init = self.emission_f_probability_sequence(o_f_matrix)

        # print('A init\n', self.A_init)
        # print('O init\n', self.O_init)
        # print('O_f init\n', self.O_f_init)

        self.A, self.O, self.O_f = np.copy(self.A_init), np.copy(self.O_init), np.copy(self.O_f_init)  # take copies, as we modify them

        self.alpha, self.beta = np.zeros((self.N, self.S)), np.zeros((self.N, self.S))  # initialize alpha and beta
        self.alpha_f, self.beta_f = np.zeros((self.feedback_length, self.S)), np.zeros((self.feedback_length, self.S))  # initialize alpha and beta

        self.A_ijk = dict()
        self.O_jl = dict()
        self.O_jf = dict()

    def transition_probability_sequence(self, a_ijk):
        a_ijt = np.zeros((self.N, self.state_total, self.state_total))
        for id, input_id in enumerate(self.input_lambda):
            for ti, input_time in enumerate(self.input_lambda[input_id]):
                a_ijt[input_time] = a_ijk[id]

        return a_ijt

    def emission_probability_sequence(self, o_jl):
        o_jt = np.zeros((self.N, self.state_total))

        for id, output_id in enumerate(self.output_lambda):
            for ti, output_time in enumerate(self.output_lambda[output_id]):
                o_jt[output_time] = o_jl[id]

        return o_jt

    def emission_f_probability_sequence(self, o_jf):
        o_jft = np.zeros((self.feedback_length, self.state_total))
        for i, input_f in enumerate(output_f_seq):
            o_jft[i] = o_jf[input_f]
        # for id, output_id in enumerate(self.output_f_lambda):
        #     for ti, output_time in enumerate(self.output_f_lambda[output_id]):
        #         print('output time {}, id {} output id {}'.format(output_time, id, output_id))
        #         o_jft[ti] = o_jf[id]

        return o_jft

    def forward(self):
        self.alpha = np.zeros((self.N, self.S))

        # base case
        for s in range(self.S):
            self.alpha[0, :] = self.pi * self.O[0]

        # recursive case
        for k in range(1, self.N):
            for j in range(self.S):
                for i in range(self.S):
                    self.alpha[k, j] += self.alpha[k - 1, i] * self.A[k, i, j] * self.O[k, j]
                    # if np.where(self.timestamp_feedback == k)[0].size == 0:
                    #     self.alpha[k, j] += self.alpha[k - 1, i] * self.A[k, i, j] * self.O[k, j]
                    # else:
                    #     k_f = np.where(self.timestamp_feedback == k)[0][0]
                    #     self.alpha[k, j] += self.alpha[k - 1, i] * self.A[k, i, j] * self.O[k, j] * self.O_f[k_f - 1, j]


        return max(np.sum(self.alpha[self.N - 1, :]), 10 ** -300)

    def backward(self):
        self.beta = np.zeros((self.N, self.S))

        # base case
        self.beta[self.N - 1, :] = 1

        # recursive case
        for k in range(self.N - 2, -1, -1):
            # print('k {}'.format(k))
            for i in range(self.S):
                for j in range(self.S):
                    self.beta[k, i] += self.beta[k + 1, j] * self.A[k + 1, i, j] * self.O[k + 1, j]
                    # if np.where(self.timestamp_feedback == k)[0].size == 0:
                    #     self.beta[k, i] += self.beta[k + 1, j] * self.A[k + 1, i, j] * self.O[k + 1, j]
                    # else:
                    #     k_f = np.where(self.timestamp_feedback == k)[0][0]
                    #     self.beta[k, i] += self.beta[k + 1, j] * self.A[k + 1, i, j] * self.O[k + 1, j] * self.O_f[k_f - 1, j]

        return max(np.sum(self.pi * self.O[0] * self.beta[0, :]), 10 ** -300)

    def forward_feedback(self):
        self.alpha_f = np.zeros((self.feedback_length, self.S))

        # base case
        for s in range(self.S):
            self.alpha_f[0, :] = self.pi * self.O_f[0]

        # recursive case
        for k_f in range(1, self.feedback_length):
            k = self.timestamp_feedback[self.output_f_index[k_f]]
            for j in range(self.S):
                for i in range(self.S):
                    self.alpha_f[k_f, j] += self.alpha_f[k_f - 1, i] * self.A[int(k % 250), i, j] * self.O[int(k % 250), j] * self.O_f[k_f, j]
                    # self.alpha_f[k_f, j] += self.alpha_f[k_f - 1, i] * self.A[int(k % 250), i, j] * self.O_f[k_f, j]

        return max(np.sum(self.alpha_f[self.feedback_length - 1, :]), 10 ** -300)

    def backward_feedback(self):
        self.beta_f = np.zeros((self.feedback_length, self.S))

        # base case
        self.beta_f[self.feedback_length - 1, :] = 1

        # recursive case
        for k_f in range(self.feedback_length - 2, -1, -1):
            k = self.timestamp_feedback[self.output_f_index[k_f]]
            for i in range(self.S):
                for j in range(self.S):
                    self.beta_f[k_f, i] += self.beta_f[k_f + 1, j] * self.A[int((k+1 % 250), i, j] * self.O[int(k+1 % 250) , j] * self.O_f[k_f + 1, j]
                    # self.beta_f[k_f, i] += self.beta_f[k_f + 1, j] * self.A[int(k % 250) + 1, i, j] * self.O_f[k_f + 1, j]

        return max(np.sum(self.pi * self.O[0] * self.beta_f[0, :]), 10 ** -300)

    def baum_welch(self):
        # do several steps of EM hill climbing
        for it in range(self.iterations):
            print('iteration=', it)
            self.pi_new = np.zeros_like(self.pi)
            self.h_ijt = np.zeros_like(self.A)
            self.a_ijt_new = np.zeros_like(self.A)
            self.g_jlt = np.zeros((self.N, self.S))
            self.O_new = np.zeros_like(self.O)
            self.g_jft = np.zeros((self.feedback_length, self.S))
            self.O_f_new = np.zeros_like(self.g_jft)
            self.w_jl = np.zeros((self.output_num, self.state_total)) + 10 ** -250
            self.w_jf = np.zeros((self.output_f_num, self.state_total)) + 10 ** -250

            # compute forward-backward matrices and return the normalizing factors za & zb
            za = self.forward()
            zb = self.backward()

            if self.feedback_length > 0:
                za_f = self.forward_feedback()
                zb_f = self.backward_feedback()

            # print('alpha\n', self.alpha)
            # print('za\n', za)
            # print('beta\n', self.beta)
            # print('zb\n', zb)
            # print('alpha_F\n', self.alpha_f)
            # print('beta_f\n', self.beta_f)

            assert abs(za - zb) < 1e-2, "it's badness 10000 if the marginals don't agree"

            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            self.pi_new += self.alpha[0, :] * self.beta[0, :] / za
            self.pi = self.pi_new / max(np.sum(self.pi_new), 10 ** -300)  # normalise pi_new

            for k in range(0, self.N):
                self.g_jlt[k] += self.alpha[k, :] * self.beta[k, :] / za

            if self.feedback_length > 0:
                for k_f in range(self.feedback_length):
                    k = self.timestamp_feedback[self.output_f_index[k_f]]
                    self.g_jft[k_f] += self.alpha_f[k_f, :] * self.beta_f[k_f, :] / za_f

            for k in range(1, self.N):
                for j in range(self.S):
                    for i in range(self.S):
                        self.h_ijt[k - 1, i, j] = self.alpha[k - 1, i] * self.A[k, i, j] * self.O[k, j] * self.beta[k, j] / za

            # A re-estimation
            for k, ti in enumerate(self.input_lambda.values()):
                self.h_ij_sumt = np.zeros_like(self.A[0]) + 10 ** -300
                if len(ti) > 0:
                    for ki, t in enumerate(ti):
                        self.h_ij_sumt += self.h_ijt[t]

                    self.h_ijk_new = np.transpose(np.transpose(self.h_ij_sumt) / np.sum(self.h_ij_sumt, 1))

                    for ki, t in enumerate(ti):
                        self.a_ijt_new[t] = self.h_ijk_new

            # O re-estimation
            for t1, to in enumerate(self.output_lambda):
                if len(self.output_lambda[to]) > 0:
                    self.O1_temp = np.zeros((1, self.state_total))
                    for t, ut in enumerate(self.output_lambda[to]):
                        self.O1_temp += self.g_jlt[ut]

                    self.w_jl[t1] = self.O1_temp
                else:
                    self.w_jl[t1] = np.zeros_like(self.O[0]) + 10 ** -250

            self.w_jl /= np.sum(self.w_jl, 0) + 10 ** -301

            for t1, to in enumerate(self.output_lambda):
                if len(self.output_lambda[to]) > 0:
                    for t, ut in enumerate(self.output_lambda[to]):
                        self.O_new[ut] = self.w_jl[t1]

            # O_f re-estimation
            if self.feedback_length > 0:
                for t1, to in enumerate(self.output_f_lambda):
                    if len(self.output_f_lambda[to]) > 0:
                        self.O1_f_temp = np.zeros((1, self.state_total))
                        for t, ut in enumerate(self.output_f_lambda[to]):
                            self.O1_f_temp += self.g_jft[ut]

                        self.w_jf[t1] = self.O1_f_temp
                    else:
                        self.w_jf[t1] = np.zeros_like(self.O_f[0]) + 10 ** -300

                self.w_jf /= np.sum(self.w_jf, 0) + 10 ** -300

                for t1, to in enumerate(self.output_f_lambda):
                    if len(self.output_f_lambda[to]) > 0:
                        for t, ut in enumerate(self.output_f_lambda[to]):
                            self.O_f_new[ut] = self.w_jf[t1]
            # else:
            #     self.O_f_new = np.zeros((self.N_f, 125))

            self.A, self.O, self.O_f = self.a_ijt_new, self.O_new, self.O_f_new

            # print('updated A=\n', self.A, '\n')
            # print('updated O=\n', self.O, '\n')

        for k, ti in enumerate(self.input_lambda.values()):
            if len(ti) > 0:
                self.A_ijk[k] = self.A[ti[0]]
            else:
                self.A_ijk[k] = np.zeros_like(self.A[0])

        for l, to in enumerate(self.output_lambda):
            if len(self.output_lambda[to]) > 0:
                self.O_jl[l] = self.O[self.output_lambda[to][0]]
            else:
                self.O_jl[l] = np.zeros_like(self.O[0])

        for l, to in enumerate(self.output_f_lambda):
            if len(self.output_f_lambda[to]) > 0:
                print('l', l, 'lambda to', self.output_f_lambda[to][0])
                self.O_jf[l] = self.O_f[self.output_f_lambda[to][0]]
            else:
                # self.O_jf[l] = np.zeros_like(self.O_f[0])
                self.O_jf[l] = o_f_matrix[l-1]

        print('O_jf')
        print(self.O_jf)
        print()

        return self.pi, self.A, self.O, self.A_ijk, self.O_jl, self.O_jf


if __name__ == "__main__":
    prob_transition = CRBM.CRBM('transition')
    prob_emission = CRBM.CRBM('emission')
    prob_emission_f = CRBM.CRBM('emission_f')

    a_matrix = prob_transition.total_transition_probs()
    o_matrix = prob_emission._o_jk()
    o_f_matrix = prob_emission_f._o_jf()

    # Training Data Import
    TrainingD = trainingData.TrainingData()
    [training_input_seq, training_output_seq] = TrainingD.io_sequence_generator()
    training_total_len = len(training_input_seq)

    def to(s, b):
        """
        Converts given number s, from base b to base 10
        s -- string representation of number
        b -- base of given number
        """
        assert (1 < b < 37)
        return int(s, b)

    feedback_seq = list()

    for i, input_value in enumerate(training_input_seq):
        input_string = str(int((input_value[0]-0.1)/2))+str(int((input_value[1]-0.1)/2))+str(int((input_value[2]-0.1)/2))
        value_base_10 = to(input_string, 5)
        feedback_seq.append(value_base_10)
    # print('feedback seq {}'.format(feedback_seq))

    # feedback sequence sample
    feedback_tperiod = 20
    training_output_f_seq = np.zeros((int(len(training_input_seq)//feedback_tperiod)+1,))

    output_f_time_stamp = np.zeros_like(training_output_f_seq)   # feedback time stamps

    for i in np.arange(0, len(training_input_seq), feedback_tperiod):
        training_output_f_seq[int(i / feedback_tperiod)] = int(feedback_seq[i])
        output_f_time_stamp[int(i / feedback_tperiod)] = int(i)  # Every 10 time-steps the feedback is received.

    # print('training_output_f_seq', training_output_f_seq)
    # print('output_f_time_stamp', output_f_time_stamp)

    pi_trained_list, A_trained_list, O_trained_list, A_ijk_list, O_jl_list, O_jf_list = list(), list(), list(), list(), list(), list()

    session_len = 250
    print('total training length --------> {}'.format(training_total_len))
    for chunk_i in range(training_total_len // session_len + 1):
        print('chunk begining', time.clock())
        print(chunk_i * session_len, min(chunk_i * session_len + session_len, training_total_len))

        input_seq = np.copy(training_input_seq[chunk_i * session_len: min(chunk_i * session_len + session_len, training_total_len)])
        output_seq = np.copy(training_output_seq[chunk_i * session_len: min(chunk_i * session_len + session_len, training_total_len)])

        a = np.where(output_f_time_stamp > chunk_i * session_len)[0]
        b = np.where(output_f_time_stamp <= chunk_i * session_len + session_len)[0]

        output_f_index = [value for value in a if value in b]
        output_f_seq = np.copy([int(training_output_f_seq[k_f]) for k_f in output_f_index])

        print('output f index: {}'.format(output_f_index))
        print('output_f_seq', output_f_seq)

        em = EM(4, input_seq, output_seq, output_f_seq, a_matrix, o_matrix, o_f_matrix, output_f_time_stamp, output_f_index)

        pi_trained, A_trained, O_trained, A_ijk, O_jl, O_jf = em.baum_welch()

        pi_trained_list.append(pi_trained)
        A_trained_list.append(A_trained)
        O_trained_list.append(O_trained)
        A_ijk_list.append(A_ijk)
        O_jl_list.append(O_jl)
        O_jf_list.append(O_jf)

        del em

        print('chunk finish', time.clock())

    # A_average = np.zeros((1000, 125, 125))
    # O_average = np.zeros((4, 125))
    #
    # for k in range(1000):
    #     A_avg_tmp = np.zeros((125, 125))
    #     i_count = 0
    #     for i in range(len(A_ijk_list)):
    #         if np.sum(A_ijk_list[i][k]) > 0:
    #             A_avg_tmp += A_ijk_list[i][k]
    #             i_count += 1
    #     A_average[k] = A_avg_tmp / i_count
    #
    # for j in range(4):
    #     O_avg_temp = np.zeros((125,))
    #     o_count = 0
    #     for i in range(len(O_jl_list)):
    #         if np.sum(O_jl_list[i][j]) > 0:
    #             O_avg_temp += O_jl_list[i][j]
    #             o_count += 1
    #     O_average[j] = O_avg_temp / o_count

    A_average = np.zeros((1000, 125, 125))
    O_average = np.zeros((4, 125))
    O_f_average = np.zeros((125, 125))

    for k in range(1000):
        A_avg_tmp = np.zeros((125, 125))
        i_count = 0
        for i in range(len(A_ijk_list)):
            if np.sum(A_ijk_list[i][k]) > 0:
                A_avg_tmp += A_ijk_list[i][k] * (i + 1)
                # i_count += 1
                i_count += i + 1
        A_average[k] = A_avg_tmp / i_count

    for j in range(4):
        O_avg_temp = np.zeros((125,))
        o_count = 0
        for i in range(len(O_jl_list)):
            if np.sum(O_jl_list[i][j]) > 0:
                O_avg_temp += O_jl_list[i][j] * (i + 1)
                o_count += i + 1
        O_average[j] = O_avg_temp / o_count

    for j in range(125):
        O_f_avg_temp = np.zeros((125,))
        o_f_count = 0

        for i in range(len(O_jf_list)):
            # print('j {} i {} '.format(j, i))
            if np.sum(O_jf_list[i][j]) > 0:
                O_f_avg_temp += O_jf_list[i][j] * (i + 1)
                o_f_count += i + 1

        if o_f_count > 0:
            O_f_average[j] = O_f_avg_temp / o_f_count

    # Save the learned data
    with open('A_average_training_data_5.pickle', 'wb') as f_a:
        pickle.dump(A_average, f_a)

    with open('O_average_training_data_5.pickle', 'wb') as f_o:
        pickle.dump(O_average, f_o)

    with open('O_f_average_training_data_5.pickle', 'wb') as f_of:
        pickle.dump(O_f_average, f_of)

