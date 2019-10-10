import numpy as np
import xlrd

np.set_printoptions(linewidth=600)
np.set_printoptions(precision=3, edgeitems=25)

agent_num = 3
state_scale = 3
input_num = 3
output_num = 2
state_total = state_scale ** agent_num
input_tot = input_num ** agent_num
output_tot = output_num ** agent_num

state_vec = np.arange(1, state_total + 1).reshape((1, state_total))

workbook_weights = xlrd.open_workbook('mlogistic_weights_0.xlsx')
worksheet_weights = workbook_weights.sheet_by_index(0)

weights_read = np.zeros((5, state_total))

for i in range(5):
    weights_read[i, :] = worksheet_weights.col_values(i)

weights_matrix = np.transpose(weights_read)

workbook_inputs = xlrd.open_workbook('mlogistic_inputs.xlsx')
worksheet_inputs = workbook_inputs.sheet_by_index(0)

inputs_read = np.zeros((agent_num, input_tot))

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

    print(A_ijt)
    return A_ijt





mlogit_transition(weights_matrix, inputs_matrix)