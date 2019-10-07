# -*- coding:   utf-8 -*-
# @author: Maziar Fooladi Mahani

# This code calculates online filtering inference for Multi-Agent Systems

# from scipy.io import loadmat
# import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import  EM_Factorial_IOHMM_backup as EM

np.set_printoptions(linewidth=520)
np.set_printoptions(precision=3, edgeitems=5)

# constants

input_num = 3
output_num = 8
agent_num = 3
state_num = 8
input_tot = agent_num ** input_num

# input sequence
# u1 = np.random.randint(1, 4, (test_session_len,))
# u2 = np.random.randint(1, 4, (test_session_len,))
# u3 = np.random.randint(1, 4, (test_session_len,))
workbook = xlrd.open_workbook('IO_sample7.xlsx')
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

# print('u dict=', u_dict)


def i_index_func(data_i):
    input_index = []
    for i_index, (i1, i2, i3) in enumerate(data_i):
        input_index.append(u_dict[i1, i2, i3])

    print('input index:')
    print(input_index)
    return input_index

def io_index_func(i_index, y_data):
    io_index = dict()
    
    for i, io_id in enumerate(zip(i_index, y_data)):
        io_index[i] = list(io_id)
    
    print('io_index')
    print(io_index)
    return io_index

input_sequence = i_index_func(data_input)

pi_trained, A_trained, O_trained, A_ijk, O_jlk = EM.baum_welch(EM.output_seq, EM.pi, 10, EM.input_seq, EM.w_transition, EM.w_observation)



# transition joint probability P(u_n, s_n-1, s_n)
# A_iju = dict()
# for k in range(input_tot):
#     A_iju[k] = np.ones((8, 8)) / 8



# output sequence
output_sequence = np.random.randint(0, 8, len(data_input))
# print('output_sequence')
# print(output_sequence)

io_sequence = io_index_func(input_sequence, data_output)

y = np.ones((len(data_input),))  # y in {1,2,3,4,5,6,7,8}


# observation joint probability P(u_n, y_n, s_n)
# O_jlk = dict()
#
# for k in range(input_tot):
#     for l in range(output_num):
#         O_jlk[k, l] = np.ones((8,)) / 8
#
# print('O_jlk', O_jlk)

# initial belief
bel0 = np.ones((state_num,))
# bel0 = np.ones((state_num,)) * np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1])

belief = np.zeros((len(data_input) + 1, state_num))

belief[0] = bel0

for t in range(len(data_input)):
    # print('t=', t)
    print('A_ijk[input_sequence[{}]]'.format(t))
    print(A_ijk[input_sequence[t]])
    # print('belief[{}]'.format(t))
    # print(belief[t])
    # print('A_iju[input_sequence[{}]] * np.transpose(belief[{}])'.format(t, t))
    # print(A_iju[input_sequence[t]] * np.transpose(belief[t]))
    # print('np.sum(A_iju[input_sequence[t]] * np.transpose(belief[t]),0)')
    # print(np.sum(A_iju[input_sequence[t]] * np.transpose(belief[t]),0))
    # print('input_sequence[{}]'.format(t))
    # print(input_sequence[t])
    # print('output_sequence[{}]'.format(t))
    # print(output_sequence[t])
    print('O_jlk[{}, {}]'.format(io_sequence[t][0], io_sequence[t][1]))
    print(O_jlk[io_sequence[t][0], io_sequence[t][1]])
    # print('np.sum(A_iju[input_sequence[t]] * np.transpose(belief[t]),0) * O_jy[input_sequence[t], output_sequence[t]]')
    # print(np.sum(A_iju[input_sequence[t]] * np.transpose(belief[t]),0) * O_jy[input_sequence[t], output_sequence[t]])
    # print(input_sequence[t], io_sequence[t])
    belief_temp = np.multiply(np.sum(A_ijk[input_sequence[t]] * np.transpose(belief[t]), 0), O_jlk[io_sequence[t][0], io_sequence[t][1]])
    # belief_temp = np.multiply(np.sum(A_ijk[input_sequence[t]] * belief[t], 0) , O_jlk[io_sequence[t][0], io_sequence[t][1]])
    belief_temp /= np.sum(belief_temp)
    belief[t+1] = np.copy(belief_temp)
    
    # print('belief_temp')
    # print(belief_temp)
print('belief')
print(belief)


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

xpos =[]
ypos =[]
zpos =[]
dz = []
for t in range(len(data_input)):
    for s in range(state_num):
        xpos.append(t)
        ypos.append(s)
        zpos.append(0)
        dz.append(belief[t, s])


num_elements = len(xpos)
dx = np.ones(1)
dy = np.ones(1)

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
plt.show()



### plot the data ############################################################
# X, Y = np.meshgrid(np.arange(0, K, 1), vec_t)
# Z1 = np.asarray(Bel_T1)
# Z2 = np.asarray(Bel_T2)
# max_bel = 0.
# min_bel = 0.
# for i in range(len(Z1)):
#     if max_bel < max(Z1[i]):
#         max_bel = max(Z1[i])
#     if min_bel > min(Z1[i]):
#         min_bel = min(Z1[i])
# data1 = Z1
# data2 = Z2
# ypos1, xpos1 = np.indices(data1.shape)
# ypos2, xpos2 = np.indices(data2.shape)
# xpos1 = xpos1.flatten()
# xpos2 = xpos2.flatten()
# ypos1 = ypos1.flatten()
# ypos2 = ypos2.flatten()
# zpos1 = np.zeros(xpos1.shape)
# zpos2 = np.zeros(xpos2.shape)
#
# fig1 = plt.figure(1)
# ax = Axes3D(fig1)
# ax.view_init(elev=90., azim=-90.)
# colors = plt.cm.jet(data1.flatten() / float(data1.max()))
# ax.bar3d(xpos1, ypos1, zpos1, .8, .8, data1.flatten(), color=colors)
# ax.set_xlabel('Time')
# ax.set_ylabel('Trust')
#
# fig2 = plt.figure(2)
# ax2 = Axes3D(fig2)
# ax2.view_init(elev=90., azim=-90.)
# colors = plt.cm.jet(data2.flatten() / float(data2.max()))
# ax2.bar3d(xpos2, ypos2, zpos2, .8, .8, data2.flatten(), color=colors)
# ax2.set_xlabel('Trust')
# ax2.set_ylabel('Time')
#
# fig3, (axp1, axp2, axvar1, axvar2) = plt.subplots(4, 1)
# axp1.plot(np.arange(K), p1)
# axp2.plot(np.arange(K), p2)
# axvar1.plot(np.arange(K), MDS1)
# axvar2.plot(np.arange(K), MDS2)
# plt.show()