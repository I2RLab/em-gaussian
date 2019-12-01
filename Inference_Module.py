# -*- coding:   utf-8 -*-
# @author: Maziar Fooladi Mahani

# This code calculates online filtering inference for Multi-Agent Systems

import numpy as np
import sys
import CRBM
import Training_Dataset_Generator as trainingData
import Test_Dataset_Generator as testdata
import EM_Module
from mayavi.mlab import *
import time
from scipy import *
import matplotlib.pyplot as plt
import pickle

print('start', time.clock())

sys.path.insert(0, '../Categorical_Boltzmann_Machines')

np.set_printoptions(linewidth=700)
np.set_printoptions(precision=3, edgeitems=5)

prob_transition = CRBM.CRBM('transition')
prob_emission = CRBM.CRBM('emission')

a_matrix = prob_transition.total_transition_probs()
o_matrix = prob_emission._o_jk()

TestD = testdata.TrainingData()
[test_input_seq, test_output_seq, test_output_f_seq] = TestD.io_sequence_generator()


'''
for chunk_i in range(training_total_len // session_len + 1):
    print(chunk_i * session_len, min(chunk_i * session_len + session_len, training_total_len))
    input_seq = np.copy(training_input_seq[chunk_i * session_len: min(chunk_i * session_len + session_len, training_total_len)])
    output_seq = np.copy(training_output_seq[chunk_i * session_len: min(chunk_i * session_len + session_len, training_total_len)])

    em = EM_Module.EM(1, input_seq, output_seq, a_matrix, o_matrix)

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

# Save the learned data
with open('A_average.pickle', 'wb') as f_a:
    pickle.dump(A_average, f_a)

with open('O_average.pickle', 'wb') as f_o:
    pickle.dump(O_average, f_o)

print(time.clock())
print(time.clock())
'''

with open('A_it3_f10_1.pickle', 'rb') as f_a:
    A_average = pickle.load(f_a)
#
with open('O_it3_f10_1.pickle', 'rb') as f_o:
    O_average = pickle.load(f_o)

with open('O_f_it3_f10_1.pickle', 'rb') as f_of:
    O_f_average = pickle.load(f_of)

# for f, pf in enumerate(O_f_average):
#     print(pf,'\n')

# print('O_average')
# print(O_average)
#
prob_emission_f = CRBM.CRBM('emission')
O_average_0 = prob_emission_f._o_jk()

O_average[0] = O_average_0[0]



# barchart([O_f_average[24], O_f_average[104], O_f_average[120], O_f_average[124]])
# barchart(O_average)
# show()

# constants
input_num = 10
output_num = 4
agent_num = 3
state_num = 125
input_tot = agent_num ** input_num

# i/o sequence
data_input = test_input_seq
data_output = test_output_seq
data_output_f = test_output_f_seq

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

belief_filtered_no_yl = np.zeros((len(data_input) + 1, state_num))
belief_filtered_no_yl[0] = 1/125

prob_yl = np.zeros((len(data_input), 4))

# belief_smoothed = np.zeros((len(data_input), state_num))
# belief_smoothed[-1] = np.ones((state_num,)) * 0.008

# prob_yl = np.zeros((len(data_input), 4))

# belief_bar_history = np.zeros((len(data_input), state_num))   # bel_bar_j --> sum(bel_bar_ij) over 'i' --> previous time step

'''
O_subjective = np.zeros_like(O_average)

for i in range(4):
    print(np.where(O_average[i] == np.max(O_average, 1)[i])[0])
    O_subjective[i, np.where(O_average[i] == np.max(O_average, 1)[i])[0]] = 1
'''

# compute filtered belief
for t in range(len(data_input)):
    # belief_bar_ij = np.multiply(np.sum(A_average[input_sequence[t]] * belief_filtered[t], 1), O_average[int(data_output[t] - 1)])
    belief_bar_ij = np.multiply(np.multiply(np.sum(A_average[input_sequence[t]] * belief_filtered[t], 1), O_average[int(data_output[t] - 1)]), O_f_average[int(data_output_f[t])])
    # belief_bar_history[t] = np.copy(belief_bar_ij)
    belief_bar_ij /= np.sum(belief_bar_ij)
    belief_filtered[t + 1] = np.copy(belief_bar_ij)

    # belief_bar_ij_no_yl = np.sum(np.multiply(A_average[input_sequence[t]], belief_filtered_no_yl[t]), 1)
    # belief_bar_ij_no_yl = np.multiply(np.sum(A_average[input_sequence[t]] * belief_filtered[t], 1), O_subjective[int(data_output[t] - 1)])
    # belief_bar_ij_no_yl = np.multiply(np.sum(A_average[input_sequence[t]] * belief_filtered[t], 1), O_average[int(data_output[t] - 1)])
    # belief_bar_ij_no_yl = np.multiply(np.sum(A_average[input_sequence[t]] * belief_filtered[t], 1), O_f_average[int(data_output_f[t])])
    belief_bar_ij_no_yl = np.multiply(np.sum(A_average[input_sequence[t]] * belief_filtered_no_yl[t], 1), O_f_average[int(data_output_f[t])])

    for k in range(4):
        prob_yl[t, k] = np.sum(np.multiply(belief_bar_ij_no_yl, O_average[k])) / np.sum(belief_bar_ij_no_yl)

    belief_filtered_no_yl[t + 1] = belief_bar_ij_no_yl / np.sum(belief_bar_ij_no_yl)

# print('belief_filtered')
# print(belief_filtered)

print('Prob(yl)_prediction:\n')
print(prob_yl)


matched_output_num = 0

for i in range(len(prob_yl)):
    print('prob output', np.where(prob_yl[i] == np.max(prob_yl[i]))[0][0] + 1)
    print('data ', data_output[i][0])
    if (np.where(prob_yl[i] == np.max(prob_yl[i]))[0][0] + 1) == data_output[i][0]:

        matched_output_num += 1

print('output prediction accuracy is %{}\n'.format(matched_output_num/len(prob_yl)*100))


print('belief filtered no output')
print(belief_filtered_no_yl)

'''
# compute smoothed belief
for t in reversed(range(1, len(data_input))):
    print('t', t)
    bel_s_temp = np.multiply(np.multiply(A_average[input_sequence[t]], belief_filtered[t]), O_average[int(data_output[t] - 1)]) / belief_bar_history[t]
    bel_s_temp *= belief_smoothed[t]
    belief_smoothed[t - 1] = np.sum(bel_s_temp, 1)

print('smoothed belief')
print(belief_smoothed)
print('\n\n\n')
'''

# plot filtered belief
figure(1)
barchart(belief_filtered[1:])
# barchart(O_average)
# barchart(belief_smoothed)
# barchart(belief_filtered_no_yl[1:])
xlabel('Time')
ylabel('Trust')
zlabel('Belief')

figure(2)
barchart(prob_yl[1:])

print('finish', time.clock())

show()

print()
