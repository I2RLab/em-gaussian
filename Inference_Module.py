# -*- coding:   utf-8 -*-
# @author: Maziar Fooladi Mahani

# This code calculates online filtering inference for Multi-Agent Systems

import numpy as np
import sys
import CRBM
import Training_Dataset_Generator as tdg
import EM_Module
from mayavi.mlab import *
import time
import xlsxwriter
import pandas as pd


print(time.clock())

sys.path.insert(0, '../Categorical_Boltzmann_Machines')

np.set_printoptions(linewidth=700)
np.set_printoptions(precision=4, edgeitems=5)

prob_transition = CRBM.CRBM('transition')
prob_emission = CRBM.CRBM('emission')

a_matrix = prob_transition.total_transition_probs()
o_matrix = prob_emission._o_jk()

TD_class = tdg.TrainingData()
[input_seq_all, output_seq_all] = TD_class.io_sequence_generator()

training_total_len = len(input_seq_all)
# training_total_len = 80

pi_trained_list, A_trained_list, O_trained_list, A_ijk_list, O_jl_list = list(), list(), list(), list(), list()

session_len = 250

em_init = EM_Module.EM(1, input_seq_all, output_seq_all, a_matrix, o_matrix)
A_init = em_init.A_init
O_init = em_init.O_init

del em_init


for i_set in range(training_total_len // session_len + 1):
    print(i_set * session_len, min(i_set * session_len + session_len, training_total_len))
    input_seq = np.copy(input_seq_all[i_set * session_len: min(i_set * session_len + session_len, training_total_len)])
    output_seq = np.copy(output_seq_all[i_set * session_len: min(i_set * session_len + session_len, training_total_len)])

    em = EM_Module.EM(10, input_seq, output_seq, a_matrix, o_matrix)

    pi_trained, A_trained, O_trained, A_ijk, O_jl = em.baum_welch()

    pi_trained_list.append(pi_trained)
    A_trained_list.append(A_trained)
    O_trained_list.append(O_trained)
    A_ijk_list.append(A_ijk)
    O_jl_list.append(O_jl)
    total_input = em.input_tot
    total_ouput = em.output_num
    del em

A_average = np.zeros((1000, 125, 125))

for k in range(total_input):
    nonzero_num = 0
    for i_set in range(len(A_ijk_list)):
        if np.sum(A_ijk_list[i_set][k]) > 0:
            A_average[k, :, :] += A_ijk_list[i_set][k]
            nonzero_num += 1
    if nonzero_num != 0:
        A_average[k] /= nonzero_num
    else:
        A_average[k] = A_init[k]

O_average = np.zeros((4, 125))

for l in range(4):
    nonzero_num = 0
    for o_set in range(len(O_jl_list)):
        if np.sum(O_jl_list[o_set][l]) > 0:
            O_average[l, :] += O_jl_list[o_set][l]
            nonzero_num += 1
    if nonzero_num != 0:
        O_average[l] /= nonzero_num
    else:
        O_average[l] = O_init[l]


# A_save = A_average.reshape(125000, 125)
# df_A = pd.DataFrame(A_save)
# filepath_A = 'learned_A_mat.xlsx'
# df_A.to_excel(filepath_A, index=False)
#
# O_save = O_average
# df_O = pd.DataFrame(O_save)
# filepath_O = 'learned_O_mat.xlsx'
# df_O.to_excel(filepath_O, index=False)

##############################
# constants
input_num = 10
output_num = 4
agent_num = 3
state_num = 125
input_tot = agent_num ** input_num

# i/o sequence
data_input = input_seq_all[0:80]
data_output = output_seq_all[0:80]

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

# initial belief
bel0 = np.ones((state_num,))

belief = np.zeros((len(data_input) + 1, state_num))

belief[0] = bel0

for t in range(len(data_input)):
    belief_temp = np.multiply(np.sum(A_average[input_sequence[t]] * belief[t], 1), O_average[data_output[t] - 1])
    belief_temp /= np.sum(belief_temp)
    belief[t + 1] = np.copy(belief_temp)

print('belief')
print(belief)

x_pos = []
y_pos = []
z_pos = []
dz = []

for t in range(1, len(data_input)):
    for s in range(state_num):
        x_pos.append(t)
        y_pos.append(s + 1)
        z_pos.append(0)
        dz.append(belief[t, s])

num_elements = len(x_pos)
dx = np.ones(1)
dy = np.ones(1)

barchart(belief[1:] * 10)

print(time.clock())

show()

input('waiting ....')