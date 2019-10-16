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
output_total = output_num ** agent_num

state_vec = np.arange(1, state_total + 1).reshape((1, state_total))

workbook_weights = xlrd.open_workbook('mlogistic_weights_0.xlsx')
worksheet_weights_transition = workbook_weights.sheet_by_index(0)
worksheet_weights_observation = workbook_weights.sheet_by_index(1)

weights_transition_read = np.zeros((5, state_total))
weights_observation_read = np.zeros((2, state_total))

for i in range(5):
    weights_transition_read[i, :] = worksheet_weights_transition.col_values(i)

weights_transition_matrix = np.transpose(weights_transition_read)

for i in range(2):
    weights_observation_read[i, :] = worksheet_weights_observation.col_values(i)

weights_observation_matrix = np.transpose(weights_observation_read)

workbook_inputs = xlrd.open_workbook('mlogistic_inputs.xlsx')
worksheet_inputs = workbook_inputs.sheet_by_index(0)

inputs_read = np.zeros((agent_num, input_total))

for i in range(agent_num):
    inputs_read[i, :] = worksheet_inputs.col_values(i)

inputs_matrix = np.transpose(inputs_read)

print('inputs_matrix')
print(inputs_matrix)


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


def mlogit_observation(w, s):
    z_matrix = np.concatenate((np.ones((1, state_total)), state_vec))
    a = np.ones((state_total, 3))
    E_matrix = []

    for t, u_t in enumerate(input):
        try:
            b = np.multiply(a, input[t])
            e_matrix = np.concatenate((z_matrix, np.transpose(b)))
            E_matrix.append(np.transpose(e_matrix))

        except:
            pass

    b_jt = np.ones((1, state_total))

    for t in range(len(E_matrix)):
        b_ij = np.empty((1, state_total))

        for iw, w_m in enumerate(w):
            beta = list()
            for ix, x in enumerate(E_matrix[t]):
                beta.append(np.exp(np.matmul(w_m, x)))  # w_l = [w_lb, w_ls, w_l1, w_l2, w_l3] & x = [1, S(t), u1, u2, u3]

            den = 1 + sum(beta[0:-1])
            beta /= den
            beta[-1] = 1. / den

            b_ij = np.concatenate((b_ij, np.array(beta).reshape(1, state_total)))
        b_ij = b_ij[1::]
        b_ij = np.transpose(b_ij)
        b_jt = np.concatenate((b_jt, b_ij[[int(output[t][0]) - 1]]))

    b_jt = b_jt[1::]

    return b_jt


transition_probability = mlogit_transition(weights_transition_matrix, inputs_matrix)

for i in range(len(transition_probability)):
    print('input=%2d max= %4.3f, min= %4.3f' % (i+1, np.max(transition_probability[i]), np.min(transition_probability[i])))


# PLOT RESULTS #
# for i in range(len(transition_probability)):
for i in range(27):
    fig = plt.figure(i)
    ax1 = fig.add_subplot(111, projection='3d')

    xpos = []
    ypos = []
    zpos = []
    dz = []

    for s0 in range(state_total):
        for s1 in range(state_total):
            xpos.append(s0 + 1)
            ypos.append(s1 + 1)
            zpos.append(0)
            dz.append(transition_probability[i, s0, s1])

    num_elements = len(xpos)
    dx = np.ones(1)
    dy = np.ones(1)

    colors = plt.cm.jet((np.asanyarray(dz).flatten()) / (float(np.asanyarray(dz).max())))

    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

# plt.show()