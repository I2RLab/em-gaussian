import numpy as np
import xlrd
import sys
import CRBM
import training_dataset_generator as tdg

sys.path.insert(0, '../Categorical_Boltzmann_Machines')

np.set_printoptions(linewidth=600)
np.set_printoptions(precision=4, edgeitems=25)

prob_transition = CRBM.CRBM('transition')
prob_emission = CRBM.CRBM('emission')

a_matrix = prob_transition.total_transition_probs()
o_matrix = prob_emission._o_jk()

prob_emission = CRBM.CRBM('emission')

TD_class = tdg.TrainingData()
[input_seq, output_seq] = TD_class.io_sequence_generator()

workbook = xlrd.open_workbook('IO_s5.xlsx')
worksheet = workbook.sheet_by_index(0)


class EM:
    def __init__(self, iterations, i_seq, o_seq):
        self.iterations = iterations

        self.state_scale = 5

        self.agent_num = 3

        self.input_num = 10

        self.output_num = 4

        self.state_total = self.state_scale ** self.agent_num

        self.input_tot = self.input_num ** self.agent_num

        self.state_vec = np.arange(1, self.state_total + 1).reshape((1, self.state_total))

        self.pi = np.ones((self.state_total,)) / self.state_total  # initial distribution

        self.data_sample_i = list()

        for i in range(self.agent_num):
            self.data_sample_i.append(worksheet.col_values(i))

        self.data_input = np.transpose(np.array(self.data_sample_i))
        self.input_seq = i_seq
        self.data_output = np.transpose(np.array(worksheet.col_values(4))).reshape((len(worksheet.col_values(4)), 1))
        self.output_seq = o_seq

        self.output_lambda = dict()  # output_lambda(t) = 1 when the t'th output is l

        self.time_seq = np.arange(len(input_seq))  # time sequence
        self.time_length = len(self.time_seq)

        for i in range(1, self.output_num + 1):
            self.output_lambda[i] = np.where(self.output_seq == i)[0]

        self.N = self.time_length
        self.S = self.pi.shape[0]

        self.input_k = dict()

        for i1 in range(1, self.input_num + 1):
            for i2 in range(1, self.input_num + 1):
                for i3 in range(1, self.input_num + 1):
                    self.input_k[i1, i2, i3] = []

                    for k in range(self.time_length):
                        if list(self.input_seq[k]) == [i1, i2, i3]:
                            self.input_k[i1, i2, i3].append(1)

                        else:
                            self.input_k[i1, i2, i3].append(0)

        self.sigma_k = self.input_k

        self.input_lambda = dict()

        self.array = np.array(list(self.sigma_k.values()))

        for u, t in enumerate(self.array):
            self.input_lambda[u] = list(np.where(t == 1)[0])

        self.io_lambda = dict()

        for i, ti in enumerate(self.input_lambda):
            for o, to in enumerate(self.output_lambda):
                self.io_lambda[ti, to] = list(set(self.input_lambda[ti]).intersection(self.output_lambda[to]))

        self.A_init = self.tranition_probability_sequence(a_matrix)
        self.O_init = self.emission_probability_sequence(o_matrix)
        print('A init\n', self.A_init)
        print('O init\n', self.O_init)

        self.alpha = np.zeros((self.N, self.S))
        self.za = 0
        self.beta = np.zeros((self.N, self.S))
        self.zb = 0

    def tranition_probability_sequence(self, a_ijk):
        a_ijt = np.zeros((self.time_length, self.state_total, self.state_total))
        for id, input_id in enumerate(self.input_lambda):
            for ti, input_time in enumerate(self.input_lambda[input_id]):
                a_ijt[input_time] = a_ijk[id]

        return a_ijt

    def emission_probability_sequence(self, o_jl):
        o_jt = np.zeros((self.time_length, self.state_total))

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
        self.A, self.O = np.copy(self.A_init), np.copy(self.O_init)  # take copies, as we modify them
        self.S = self.pi.shape[0]
        self.obs_length = int(len(self.A))

        # do several steps of EM hill climbing
        for it in range(self.iterations):
            print('iteration=', it)
            self.pi_new = np.zeros_like(self.pi)
            self.A1 = np.zeros_like(self.A)
            self.A1_New = np.zeros_like(self.A)
            self.O1 = np.zeros((self.obs_length, self.S))

            # compute forward-backward matrices
            self.za = self.forward()
            self.zb = self.backward()
            print('alpha\n', self.alpha)
            print('za\n', self.za)
            print('beta\n', self.beta)
            print('zb\n', self.zb)

            assert abs(self.za - self.zb) < 1e-2, "it's badness 10000 if the marginals don't agree"

            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            self.pi_new += self.alpha[0, :] * self.beta[0, :] / self.za
            self.pi = self.pi_new / max(np.sum(self.pi_new), 10 ** -300)  # normalise pi_new

            for k in range(0, self.obs_length):
                self.O1[k] += self.alpha[k, :] * self.beta[k, :] / self.za

            for k in range(1, self.obs_length):
                for j in range(self.S):
                    for i in range(self.S):
                        self.A1[k - 1, i, j] = self.alpha[k - 1, i] * self.A[k, i, j] * self.O[k, j] * self.beta[k, j] / self.za

            for k, ti in enumerate(self.input_lambda.values()):
                self.H = np.zeros_like(self.A[0]) + 10 ** -300
                if len(ti) > 0:
                    for ki, t in enumerate(ti):
                        self.H += self.A1[t]

                    self.H_temp = np.transpose(np.transpose(self.H) / np.sum(self.H, 1))

                    for ki, t in enumerate(ti):
                        self.A1_New[t] = self.H_temp

            self.O_New = np.zeros_like(self.O)
            self.w_jl = np.zeros((self.output_num, self.state_total)) + 10 ** -250

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

            self.A, self.O = self.A1_New, self.O_New
            # print('A=\n', A, '\n')
            # print('O=\n', O, '\n')

        self.A_ijk = dict()

        for k, ti in enumerate(self.input_lambda.values()):
            if len(ti) > 0:
                self.A_ijk[k] = self.A[ti[0]]
            else:
                self.A_ijk[k] = np.zeros_like(self.A[0])

        self.O_jl = dict()

        for l, to in enumerate(self.output_lambda):
            if len(self.output_lambda[to]) > 0:
                self.O_jl[l] = self.O[self.output_lambda[to][0]]
            else:
                self.O_jl[l] = np.zeros_like(self.O[0])

        print('A=\n', self.A, '\n')
        print('O=\n', self.O, '\n')

        return self.pi, self.A, self.O, self.A_ijk, self.O_jl


if __name__ == "__main__":
    em = EM(5, input_seq, output_seq)
    pi_trained, A_trained, O_trained, A_ijk, O_jl = em.baum_welch()