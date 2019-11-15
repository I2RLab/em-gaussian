# -*- coding:   utf-8 -*-
# @author: Maziar Fooladi Mahani

# This code calculates online filtering inference for Multi-Agent Systems

import numpy as np
import xlrd
import EM_Factorial_IOHMM_backup as EM
# import EM_S5 as EM
import EM_Module as EM
from mayavi.mlab import *

np.set_printoptions(linewidth=520)
np.set_printoptions(precision=3, edgeitems=15)

# constants
input_num = 10
output_num = 4
agent_num = 3
state_num = 125
input_tot = agent_num ** input_num

# input sequence
workbook = xlrd.open_workbook('IO_s5_test1.xlsx')
worksheet = workbook.sheet_by_index(0)

data_read = list()

for i in range(agent_num):
    data_read.append(worksheet.col_values(i))

data_input = np.transpose(np.array(data_read))

data_output = np.transpose(np.array(worksheet.col_values(5))).reshape((len(worksheet.col_values(5)),))

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


def io_index_func(i_index, y_data):
    io_index = dict()
    
    for i, io_id in enumerate(zip(i_index, y_data)):
        io_index[i] = list(io_id)
    
    return io_index


input_sequence = i_index_func(data_input)

# pi_trained, A_trained, O_trained, A_ijk, O_jlk = EM.baum_welch(EM.output_seq, EM.pi, 7, EM.input_seq, EM.w_transition, EM.w_observation)
pi_trained, A_trained, O_trained, A_ijk, O_jl = EM.baum_welch(EM.output_seq, EM.pi, 6, EM.input_seq)

# io_sequence = io_index_func(input_sequence, data_output)

# y = np.ones((len(data_input),))  # y = {1,2,3,4}

# initial belief_filtered
bel0 = np.ones((state_num,))

belief = np.zeros((len(data_input) + 1, state_num))

belief[0] = bel0

for t in range(len(data_input)):
    belief_temp = np.multiply(np.sum(A_ijk[input_sequence[t]] * belief[t], 1), O_jl[data_output[t]])
    belief_temp /= np.sum(belief_temp)
    belief[t+1] = np.copy(belief_temp)

print('belief_filtered')
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
show()

