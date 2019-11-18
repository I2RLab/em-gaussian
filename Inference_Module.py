# -*- coding:   utf-8 -*-
# @author: Maziar Fooladi Mahani

# This code calculates online filtering inference for Multi-Agent Systems

import numpy as np
import sys
import CRBM
import Training_Dataset_Generator as trainingdata
import Test_Dataset_Generator as testdata
import EM_Module
from mayavi.mlab import *
import time
from scipy import *
import matplotlib.pyplot as plt

print(time.clock())

sys.path.insert(0, '../Categorical_Boltzmann_Machines')

np.set_printoptions(linewidth=700)
np.set_printoptions(precision=4, edgeitems=80)

prob_transition = CRBM.CRBM('transition')
prob_emission = CRBM.CRBM('emission')

a_matrix = prob_transition.total_transition_probs()
o_matrix = prob_emission._o_jk()

TrainingD = trainingdata.TrainingData()
[training_input_seq, training_output_seq] = TrainingD.io_sequence_generator()

TestD = testdata.TrainingData()
[test_input_seq, test_output_seq] = TestD.io_sequence_generator()


training_total_len = len(training_input_seq)
# training_total_len = 80


###################################
# fig, axs = plt.subplots(2)
# fig.suptitle('Input/Output Sequences')
# axs[0].plot(training_input_seq,cmap='gray')
# axs[1].plot(training_output_seq)
# plt.show()
# print()


pi_trained_list, A_trained_list, O_trained_list, A_ijk_list, O_jl_list = list(), list(), list(), list(), list()

session_len = 250

em_init = EM_Module.EM(1, training_input_seq, training_output_seq, a_matrix, o_matrix)
A_init = em_init.A_init
O_init = em_init.O_init

del em_init


for i_set in range(training_total_len // session_len + 1):
    print(i_set * session_len, min(i_set * session_len + session_len, training_total_len))
    input_seq = np.copy(training_input_seq[i_set * session_len: min(i_set * session_len + session_len, training_total_len)])
    output_seq = np.copy(training_output_seq[i_set * session_len: min(i_set * session_len + session_len, training_total_len)])

    em = EM_Module.EM(10, input_seq, output_seq, a_matrix, o_matrix)

    pi_trained, A_trained, O_trained, A_ijk, O_jl = em.baum_welch()

    # print('New A_ijk =\n {}\n'.format(A_ijk))
    # print('New O_jl =\n {}\n'.format(O_jl))

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
    A_average[k] = A_avg_tmp/i_count

for j in range(4):
    O_avg_temp = np.zeros((125,))
    o_count = 0
    for i in range(len(O_jl_list)):
        if np.sum(O_jl_list[i][j]) > 0:
            O_avg_temp += O_jl_list[i][j]
            o_count += 1
    O_average[j] = O_avg_temp/o_count


# A_average = np.ones((1000, 125, 125)) * 0.008
# A_average = np.random.random((1000, 125, 125))
# O_average = np.ones((4, 125)) * 0.008

# constants
input_num = 10
output_num = 4
agent_num = 3
state_num = 125
input_tot = agent_num ** input_num

# i/o sequence
data_input = test_input_seq
data_output = test_output_seq

test_session_len = len(data_input)
u_dict = dict()
input_id = 0

for i in range(1, input_num + 1):
    for j in range(1, input_num + 1):
        for k in range(1, input_num + 1):
            u_dict[i, j, k] = input_id
            input_id += 1


def i_index_func(data_i):
    input_index = []
    for i_index, (i1, i2, i3) in enumerate(data_i):
        input_index.append(u_dict[i1, i2, i3])

    return input_index


input_sequence = i_index_func(data_input)

# initial belief_filtered
bel0 = np.ones((state_num,))

belief_filtered = np.zeros((len(data_input) + 1, state_num))
belief_filtered[0] = bel0

belief_smoothed = np.zeros((len(data_input), state_num))
belief_smoothed[-1] = np.ones((state_num,)) * 0.008

prob_yl = np.zeros((len(data_input) + 1,))

belief_bar_history = np.zeros((len(data_input), state_num))  # bel_bar_j --> sum(bel_bar_ij) over 'i' --> previous time step

# compute filtered belief
for t in range(len(data_input)):
    belief_bar_ij = np.multiply(np.sum(A_average[input_sequence[t]] * belief_filtered[t], 1), O_average[int(data_output[t] - 1)])
    belief_bar_history[t] = np.copy(belief_bar_ij)
    prob_yl_bar = belief_bar_ij * O_average[int(data_output[t]) - 1]
    belief_den_temp = np.copy(np.sum(belief_bar_ij))
    belief_bar_ij /= belief_den_temp
    prob_yl[t + 1] = np.sum(prob_yl_bar) / belief_den_temp
    belief_filtered[t + 1] = np.copy(belief_bar_ij)


print('belief_filtered')
print(belief_filtered)

print('Prob yt')
print(prob_yl)

# compute smoothed belief
for t in reversed(range(1, len(data_input))):
    print('t', t)
    bel_s_temp = np.multiply(np.multiply(A_average[input_sequence[t]], belief_filtered[t]), O_average[int(data_output[t] - 1)]) / belief_bar_history[t]
    bel_s_temp *= belief_smoothed[t]
    belief_smoothed[t - 1] = np.sum(bel_s_temp, 1)

print('smoothed belief')
print(belief_smoothed)
print('\n\n\n')


# plot filtered belief
barchart(belief_filtered[1:])
# barchart(belief_smoothed)
xlabel('Time')
ylabel('Trust')
zlabel('Belief')

print(time.clock())

show()

print()