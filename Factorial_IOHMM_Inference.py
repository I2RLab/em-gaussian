# -*- coding:   utf-8 -*-
# @author: Maziar Fooladi Mahani

# This code calculates online filtering inference for Multi-Agent Systems

# from scipy.io import loadmat
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm, logistic, rv_discrete

np.set_printoptions(linewidth=520)
np.set_printoptions(precision=3, edgeitems=5)

# constants
test_session_len = 100

input_num = 6
output_num = 8
agent_num = 3
state_num = 8

# input sequence
u1 = np.ones((test_session_len,))
u2 = np.ones((test_session_len,)) * 1
u3 = np.ones((test_session_len,)) * 3
u_dict = dict()
input_index = 0

for i in range(input_num):
    for j in range(input_num):
        for k in range(input_num):
            u_dict[i, j, k] = input_index
            input_index += 1

print('u dict=', u_dict)


def input_data(u1, u2, u3):
    input_sequence = []
    for i_index, (i1, i2, i3) in enumerate(zip(u1, u2, u3)):
        input_sequence.append(u_dict[i1, i2, i3])

    print('input sequence index:')
    print(input_sequence)
    return input_sequence


input_sequence = input_data(u1, u2, u3)

# transition joint probability P(u_n, s_n-1, s_n)
A_iju = np.ones((input_num ** agent_num - 1, state_num, state_num))  # A(u_n, s_n-1, s_n_

# output sequence
output_sequence = np.random.randint(1, 9, test_session_len)
print('output_sequence')
print(output_sequence)
y = np.ones((test_session_len,))  # y in {1,2,3,4,5,6,7,8}


# observation joint probability P(u_n, y_n, s_n)
O_jy = np.random.randint(1, 8, (input_num ** agent_num - 1, output_num, state_num)) / 8 # O(u_n, y_n, s_n)

# initial belief
# bel0 = np.ones((state_num,)) / state_num
bel0 = np.ones((state_num,)) * np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1])

belief = np.zeros((test_session_len + 1, state_num))

belief[0] = bel0

for t in range(test_session_len):
    # print('A_iju[input_sequence[{}]]'.format(t))
    # print(A_iju[input_sequence[t]])
    # print('belief[t]')
    # print(belief[t])
    # print('A_iju[input_sequence[t]] * np.transpose(belief[t])')
    # print(A_iju[input_sequence[t]] * np.transpose(belief[t]))
    # print('np.sum(A_iju[input_sequence[t]] * np.transpose(belief[t]),0)')
    # print(np.sum(A_iju[input_sequence[t]] * np.transpose(belief[t]),0))
    # print('input_sequence[{}]'.format(t))
    # print(input_sequence[t])
    # print('output_sequence[{}]'.format(t))
    # print(output_sequence[t])
    # print('O_jy[input_sequence[{}], output_sequence[{}]]'.format(t, t))
    # print(O_jy[input_sequence[t], output_sequence[t]])
    # print('np.sum(A_iju[input_sequence[t]] * np.transpose(belief[t]),0) * O_jy[input_sequence[t], output_sequence[t]]')
    # print(np.sum(A_iju[input_sequence[t]] * np.transpose(belief[t]),0) * O_jy[input_sequence[t], output_sequence[t]])
    belief_temp = np.sum(A_iju[input_sequence[t]] * np.transpose(belief[t]),0) * O_jy[input_sequence[t], output_sequence[t]]
    belief_temp = belief_temp / np.sum(belief_temp)
    print('belief_temp')
    print(belief_temp)
    belief[t+1] = belief_temp
    print('belief')
    print(belief)
    print()






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
