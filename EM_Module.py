import numpy as np
import sys
import CRBM
import Training_Dataset_Generator as trainingData
import pickle

sys.path.insert(0, '../Categorical_Boltzmann_Machines')

np.set_printoptions(linewidth=700)
np.set_printoptions(precision=4, edgeitems=80)

class EM:
    def __init__(self, iterations, input_seq, output_seq, a_matrix, o_matrix):
        self.a_matrix = a_matrix
        self.o_matrix = o_matrix

        self.iterations = iterations

        self.state_scale = 5

        self.agent_num = 3

        self.input_num = 10

        self.output_num = 4

        self.state_total = self.state_scale ** self.agent_num

        self.input_tot = self.input_num ** self.agent_num

        self.state_vec = np.arange(1, self.state_total + 1).reshape((1, self.state_total))

        self.pi = np.ones((self.state_total,)) / self.state_total  # initial distribution

        self.output_lambda = dict()

        self.N = len(input_seq)

        self.S = self.pi.shape[0]

        for i in range(1, self.output_num + 1):
            self.output_lambda[i] = np.where(output_seq == i)[0]

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

        self.input_lambda = dict()

        self.array = np.array(list(self.input_k.values()))

        for u, t in enumerate(self.array):
            self.input_lambda[u] = list(np.where(t == 1)[0])

        self.A_init = self.transition_probability_sequence(a_matrix)
        self.O_init = self.emission_probability_sequence(o_matrix)

        # print('A init\n', self.A_init)
        # print('O init\n', self.O_init)

        self.A, self.O = np.copy(self.A_init), np.copy(self.O_init)  # take copies, as we modify them

        self.alpha, self.beta = np.zeros((self.N, self.S)), np.zeros((self.N, self.S))  # initialize alpha and beta

        self.A_ijk = dict()
        self.O_jl = dict()

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

    def forward(self):
        self.alpha = np.zeros((self.N, self.S))

        # base case
        for s in range(self.S):
            self.alpha[0, :] = self.pi * self.O[0]

        # recursive case
        for k in range(1, self.N):
            for j in range(self.S):
                for i in range(self.S):
                    if np.isnan(self.alpha[k - 1, i]) or np.isnan(self.A[k, i, j]) or np.isnan(self.O[k, j]):
                        pass
                    else:
                        self.alpha[k, j] += self.alpha[k - 1, i] * self.A[k, i, j] * self.O[k, j]

        return max(np.sum(self.alpha[self.N - 1, :]), 10 ** -300)

    def backward(self):
        self.beta = np.zeros((self.N, self.S))

        # base case
        self.beta[self.N - 1, :] = 1

        # recursive case
        for k in range(self.N - 2, -1, -1):
            for i in range(self.S):
                for j in range(self.S):
                    self.beta[k, i] += self.beta[k + 1, j] * self.A[k + 1, i, j] * self.O[k + 1, j]

        return max(np.sum(self.pi * self.O[0] * self.beta[0, :]), 10 ** -300)

    def baum_welch(self):
        # do several steps of EM hill climbing
        for it in range(self.iterations):
            print('iteration=', it)
            self.pi_new = np.zeros_like(self.pi)
            self.h_ijt = np.zeros_like(self.A)
            self.a_ijt_new = np.zeros_like(self.A)
            self.O1 = np.zeros((self.N, self.S))
            self.O_New = np.zeros_like(self.O)
            self.w_jl = np.zeros((self.output_num, self.state_total)) + 10 ** -250
            
            # compute forward-backward matrices and return the normalizing factors za & zb
            za = self.forward()
            zb = self.backward()

            # print('alpha\n', self.alpha)
            # print('za\n', za)
            # print('beta\n', self.beta)
            # print('zb\n', zb)

            assert abs(za - zb) < 1e-2, "it's badness 10000 if the marginals don't agree"

            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            self.pi_new += self.alpha[0, :] * self.beta[0, :] / za
            self.pi = self.pi_new / max(np.sum(self.pi_new), 10 ** -300)  # normalise pi_new

            for k in range(0, self.N):
                self.O1[k] += self.alpha[k, :] * self.beta[k, :] / za

            for k in range(1, self.N):
                for j in range(self.S):
                    for i in range(self.S):
                        self.h_ijt[k - 1, i, j] = self.alpha[k - 1, i] * self.A[k, i, j] * self.O[k, j] * self.beta[k, j] / za

            for k, ti in enumerate(self.input_lambda.values()):
                self.h_ij_sumt = np.zeros_like(self.A[0]) + 10 ** -300
                if len(ti) > 0:
                    for ki, t in enumerate(ti):
                        self.h_ij_sumt += self.h_ijt[t]

                    self.h_ijk_new = np.transpose(np.transpose(self.h_ij_sumt) / np.sum(self.h_ij_sumt, 1))

                    for ki, t in enumerate(ti):
                        self.a_ijt_new[t] = self.h_ijk_new

            for t1, to in enumerate(self.output_lambda):
                if len(self.output_lambda[to]) > 0:
                    self.O1_temp = np.zeros((1, self.state_total))
                    for t, ut in enumerate(self.output_lambda[to]):
                        self.O1_temp += self.O1[ut]

                    self.w_jl[t1] = self.O1_temp
                else:
                    self.w_jl[t1] = np.zeros_like(self.O[0]) + 10 ** -250

            self. w_jl /= np.sum(self.w_jl, 0) + 10 ** -301

            for t1, to in enumerate(self.output_lambda):
                if len(self.output_lambda[to]) > 0:
                    for t, ut in enumerate(self.output_lambda[to]):
                        self.O_New[ut] = self.w_jl[t1]

            self.A, self.O = self.a_ijt_new, self.O_New
            
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

        # print('A=\n', self.A, '\n')
        # print('O=\n', self.O, '\n')

        return self.pi, self.A, self.O, self.A_ijk, self.O_jl


if __name__ == "__main__":
    prob_transition = CRBM.CRBM('transition')
    prob_emission = CRBM.CRBM('emission')

    a_matrix = prob_transition.total_transition_probs()
    o_matrix = prob_emission._o_jk()

    TrainingD = trainingData.TrainingData()
    [training_input_seq, training_output_seq] = TrainingD.io_sequence_generator()
    training_total_len = len(training_input_seq)
    print('total length', training_total_len)
    pi_trained_list, A_trained_list, O_trained_list, A_ijk_list, O_jl_list = list(), list(), list(), list(), list()

    session_len = 250

    # em_init = EM(1, training_input_seq, training_output_seq, a_matrix, o_matrix)
    # A_init = em_init.A_init
    # O_init = em_init.O_init
    # del em_init

    for i_set in range(training_total_len // session_len + 1):
        print(i_set * session_len, min(i_set * session_len + session_len, training_total_len))

        input_seq = np.copy(training_input_seq[i_set * session_len: min(i_set * session_len + session_len, training_total_len)])
        output_seq = np.copy(training_output_seq[i_set * session_len: min(i_set * session_len + session_len, training_total_len)])

        em = EM(10, input_seq, output_seq, a_matrix, o_matrix)

        pi_trained, A_trained, O_trained, A_ijk, O_jl = em.baum_welch()

        pi_trained_list.append(pi_trained)
        A_trained_list.append(A_trained)
        O_trained_list.append(O_trained)
        A_ijk_list.append(A_ijk)
        O_jl_list.append(O_jl)

        del em

    A_average = np.zeros((1000, 125, 125))
    O_average = np.zeros((4, 125))

    for k in range(1000):
        A_avg_tmp = np.zeros((125, 125))
        i_count = 0
        for i in range(len(A_ijk_list)):
            if np.sum(A_ijk_list[i][k]) > 0:
                A_avg_tmp += A_ijk_list[i][k]
                i_count += 1
        A_average[k] = A_avg_tmp / i_count

    for j in range(4):
        O_avg_temp = np.zeros((125,))
        o_count = 0
        for i in range(len(O_jl_list)):
            if np.sum(O_jl_list[i][j]) > 0:
                O_avg_temp += O_jl_list[i][j]
                o_count += 1
        O_average[j] = O_avg_temp / o_count

    A_average = np.zeros((1000, 125, 125))
    O_average = np.zeros((4, 125))

    for k in range(1000):
        A_avg_tmp = np.zeros((125, 125))
        i_count = 0
        for i in range(len(A_ijk_list)):
            if np.sum(A_ijk_list[i][k]) > 0:
                A_avg_tmp += A_ijk_list[i][k]
                i_count += 1
        A_average[k] = A_avg_tmp / i_count

    for j in range(4):
        O_avg_temp = np.zeros((125,))
        o_count = 0
        for i in range(len(O_jl_list)):
            if np.sum(O_jl_list[i][j]) > 0:
                O_avg_temp += O_jl_list[i][j]
                o_count += 1
        O_average[j] = O_avg_temp / o_count

    # Save the learned data
    with open('A_average_EM.pickle', 'wb') as f_a:
        pickle.dump(A_average, f_a)

    with open('O_average_EM.pickle', 'wb') as f_o:
        pickle.dump(O_average, f_o)
