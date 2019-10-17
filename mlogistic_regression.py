import numpy as np
import xlrd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(linewidth=600)
np.set_printoptions(precision=3, edgeitems=25)

agent_num = 3
state_scale = 3
input_num = 3
output_num = 2
state_total = state_scale ** agent_num
input_total = input_num ** agent_num
output_total = 4

state_vec = np.arange(1, state_total + 1).reshape((1, state_total))

workbook_weights = xlrd.open_workbook('mlogistic_weights_0.xlsx')
worksheet_weights_transition = workbook_weights.sheet_by_index(0)
worksheet_weights_observation = workbook_weights.sheet_by_index(1)

weights_transition_read = np.zeros((5, state_total))
weights_observation_read = np.zeros((4, 2))

for i in range(5):
    weights_transition_read[i, :] = worksheet_weights_transition.col_values(i)

weights_observation_matrix = np.transpose(weights_transition_read)

for i in range(2):
    weights_observation_read[:, i] = worksheet_weights_observation.col_values(i)

# weights_observation_matrix = np.transpose(weights_observation_read)
weights_observation_matrix = weights_observation_read

workbook_inputs = xlrd.open_workbook('mlogistic_inputs.xlsx')

worksheet_inputs = workbook_inputs.sheet_by_index(0)
inputs_read = np.zeros((agent_num, input_total))

for i in range(agent_num):
    inputs_read[i, :] = worksheet_inputs.col_values(i)

worksheet_outputs = workbook_inputs.sheet_by_index(1)
output_read = worksheet_outputs.col_values(0)

inputs_matrix = np.transpose(inputs_read)

print('inputs_matrix')
print(inputs_matrix)

print('output sequence')
print(output_read)

def mlogit_transition(w, u):
    x_matrix = np.ones((1, state_total))
    y_matrix = state_vec
    z_matrix = np.concatenate((x_matrix, y_matrix))
    a = np.ones((state_total, agent_num))
    E_matrix = []

    for t, u_t in enumerate(u):
        try:
            c = np.multiply(a, u_t)
            e_matrix = np.concatenate((z_matrix, np.transpose(c)))
            E_matrix.append(np.transpose(e_matrix))

        except NameError:
            pass

    a_ijt = np.ones((state_total, state_total)) / state_total

    for t in range(len(E_matrix)):
        a_ij = np.empty((1, state_total))

        for ix, x in enumerate(E_matrix[t]):
            beta = list()
            for iw, w_m in enumerate(w):
                beta.append(np.exp(np.matmul(w_m, x)))

            den = 1 + sum(beta[0:-1])
            beta /= den
            beta[-1] = 1. / den

            a_ij = np.concatenate((a_ij, np.array(beta).reshape(1, state_total)))

        a_ij = a_ij[1::]

        a_ijt = np.concatenate((a_ijt, a_ij))

    A_ijt = a_ijt.reshape((int(len(a_ijt) / state_total), state_total, state_total))
    A_ijt = A_ijt[1::]

    # print(A_ijt)
    return A_ijt


def mlogit_observation(w, y):
    w_pivot = w[0:-1]
    o_ljt = np.ndarray((len(y), output_total, state_total))
    o_lj = np.ndarray((output_total, state_total))
    
    for il, l in enumerate(y):
        for s in range(state_total):
            o_lj_temp = np.exp(np.matmul(w_pivot, np.array([1, s])))
            for i in range(output_total - 1):
                o_lj[i, s] = o_lj_temp[i] / (1 + np.sum(o_lj_temp))
                
            o_lj[output_total - 1, s] = 1 / (1 + np.sum(o_lj_temp))
        
        o_ljt[il] = o_lj
        
    return o_ljt
        

 
    
 
o_ljt = mlogit_observation(weights_observation_matrix, output_read)

transition_probability = mlogit_transition(weights_transition_matrix, inputs_matrix)

for i in range(len(transition_probability)):
    print('input=%2d max= %4.3f, min= %4.3f' % (i+1, np.max(transition_probability[i]), np.min(transition_probability[i])))


# PLOT RESULTS #
# for i in range(len(transition_probability)):
# for i in range(27):
#     fig = plt.figure(i)
#     ax1 = fig.add_subplot(111, projection='3d')
#
#     xpos = []
#     ypos = []
#     zpos = []
#     dz = []
#
#     for s0 in range(state_total):
#         for s1 in range(state_total):
#             xpos.append(s0 + 1)
#             ypos.append(s1 + 1)
#             zpos.append(0)
#             dz.append(transition_probability[i, s0, s1])
#
#     num_elements = len(xpos)
#     dx = np.ones(1)
#     dy = np.ones(1)
#
#     colors = plt.cm.jet((np.asanyarray(dz).flatten()) / (float(np.asanyarray(dz).max())))
#
#     ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

# plt.show()