# -*- coding:   utf-8 -*-
# @author: Maziar Fooladi Mahani

# This code calculates online filtering inference for Multi-Agent Systems

import numpy as np
import sys
import CRBM
import Training_Dataset_Generator as trainingData
import Test_Dataset_Generator as testdata
from mayavi.mlab import *

from scipy import *
import pickle

sys.path.insert(0, '../Categorical_Boltzmann_Machines')

np.set_printoptions(linewidth=700)
np.set_printoptions(precision=3, edgeitems=25)

# prob_transition = CRBM.CRBM('transition')
# A_average = prob_transition.total_transition_probs()
# for i in arange(0, 1000, 5):
#     barchart(A_average[i])
#     view(azimuth=180, elevation=180)
#     show()
#
prob_emission = CRBM.CRBM('emission')
# O_average = prob_emission._o_jk()
#
# prob_emission_f = CRBM.CRBM('emission_f')
# O_f_average = prob_emission_f._o_jf()

# TestD = testdata.TrainingData()
# [test_input_seq, test_output_seq, test_output_f_seq] = TestD.io_sequence_generator()

TrainingD = trainingData.TrainingData()
[test_input_seq, test_output_seq, test_output_f_seq] = TrainingD.io_sequence_generator()


with open('data_info.pickle', 'rb') as d_i:
    data_info = pickle.load(d_i)

[it_str, f_str, num_str, k_gain] = data_info
print('data info {}'.format(data_info))

with open('A_it{}_f{}_{}_g{}.pickle'.format(it_str, f_str, num_str, k_gain), 'rb') as f_a:
    A_average = pickle.load(f_a)

with open('O_it{}_f{}_{}_g{}.pickle'.format(it_str, f_str, num_str, k_gain), 'rb') as f_o:
    O_average = pickle.load(f_o)

with open('O_f_it{}_f{}_{}_g{}.pickle'.format(it_str, f_str, num_str, k_gain), 'rb') as f_of:
    O_f_average = pickle.load(f_of)

# for i in arange(0, 1000, 5):
#     barchart(A_average[i])
#     view(azimuth=180, elevation=180)
#     show()
# barchart(O_f_average)
barchart(O_average)
show()
#
# constants
input_num = 10
output_num = 4
agent_num = 3
state_num = 125

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
belief_filtered_no_yl[0] = 1

belief_filtered_no_yf = np.zeros((len(data_input) + 1, state_num))
belief_filtered_no_yf[0] = 1

prob_yl = np.zeros((len(data_input), 4))
prob_yf = np.zeros((len(data_input), state_num))

# belief_smoothed = np.zeros((len(data_input), state_num))
# belief_smoothed[-1] = np.ones((state_num,)) * 0.008
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

    belief_bar_ij_no_yl = np.multiply(np.sum(A_average[input_sequence[t]] * belief_filtered_no_yl[t], 1), O_f_average[int(data_output_f[t])])
    # print('belief_bar_no_yl {} \n'.format(belief_bar_ij_no_yl))
    # belief_bar_ij_no_yf = np.multiply(np.sum(A_average[input_sequence[t]] * belief_filtered_no_yf[t], 1), O_average[int(data_output[t] - 1)])

    for k in range(4):
        prob_yl[t, k] = np.sum(np.multiply(belief_bar_ij_no_yl, O_average[k])) / np.sum(belief_bar_ij_no_yl)

    # for k in range(125):
    #     prob_yf[t, k] = np.sum(np.multiply(belief_bar_ij_no_yf, O_f_average[k])) / np.sum(belief_bar_ij_no_yf)

    belief_filtered_no_yl[t + 1] = belief_bar_ij_no_yl / np.sum(belief_bar_ij_no_yl)
    # belief_filtered_no_yf[t + 1] = belief_bar_ij_no_yf / np.sum(belief_bar_ij_no_yf)


matched_output_num = 0

for i in range(len(prob_yl)):
    print('{} U {},    Y -> {} - {} <- MLE(Y), prob(yl) {}'.format(i, data_input[i], data_output[i], np.where(prob_yl[i] == np.max(prob_yl[i]))[0][0] + 1, prob_yl[i]))
    if (np.where(prob_yl[i] == np.max(prob_yl[i]))[0][0] + 1) == data_output[i]:
        matched_output_num += 1

# matched_output_f_num = 0
#
# for i in range(len(prob_yf)):
#     if (np.where(prob_yf[i] == np.max(prob_yf[i]))[0][0]) == data_output_f[i]:
#         matched_output_f_num += 1

print('output prediction accuracy is %{}\n'.format(matched_output_num/len(prob_yl)*100))
# print('output_f prediction accuracy is %{}\n'.format(matched_output_f_num/len(prob_yf)*100))


# print('belief filtered no output')
# print(belief_filtered_no_yl)

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
bel_state = np.zeros((test_session_len, 17))
bel_state[:, 5] = bel_state[:, 11] = -.1
# bel_state[:, 12] = bel_state[:, 25] = -.1

for i in range(1, len(belief_filtered)):
    max_bel_state_1st = np.where(belief_filtered[i] == np.max(belief_filtered[i]))[0][0]
    max_bel_state_2nd = np.where(belief_filtered[i] == sorted(list(set(belief_filtered[i].flatten().tolist())))[-2])[0][0]
    max_bel_state_3rd = np.where(belief_filtered[i] == sorted(list(set(belief_filtered[i].flatten().tolist())))[-3])[0][0]
    max_bel_state_4th = np.where(belief_filtered[i] == sorted(list(set(belief_filtered[i].flatten().tolist())))[-4])[0][0]
    for r in range(3):
        bel_state[i-1, int(prob_emission.frm(max_bel_state_1st, 5)[r])+6*r] = np.max(belief_filtered[i])
        # bel_state[i-1, int(prob_emission.frm(max_bel_state_2nd, 5)[r])+6*r] = sorted(list(set(belief_filtered[i].flatten().tolist())))[-2]
        # bel_state[i-1, int(prob_emission.frm(max_bel_state_4th, 5)[r])+6*r] = sorted(list(set(belief_filtered[i].flatten().tolist())))[-3]
        # bel_state[i-1, int(prob_emission.frm(max_bel_state_3rd, 5)[r])+6*r] = sorted(list(set(belief_filtered[i].flatten().tolist())))[-4]

for i in range(test_session_len//200):
    figure(size=(1400, 300))
    barchart(bel_state[i*200:((i+1)*200)],lateral_scale=.8)
    view(0.0, 0.0, 70)

    view(azimuth=360, elevation=360)
    print(view())

    show()



# plot filtered belief
# figure(1)
# barchart(belief_filtered[1:] * 50)
# barchart(belief_filtered_no_yl[1:] * 50)

# xlabel('Time')
# ylabel('Trust')
# zlabel('Belief')
#
# figure(2)
# barchart(prob_yl[1:])
#
# show()
