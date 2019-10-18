import numpy as np
import xlrd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(linewidth=600)
np.set_printoptions(precision=2, edgeitems=25)

workbook = xlrd.open_workbook('mlogistic_states.xlsx')
worksheet = workbook.sheet_by_index(3)

states = np.ndarray((27, 3))

for i in range(3):
    states[:, i] = worksheet.col_values(i)

print('states')
print(states)

prob_ws = np.ones((27, 27))

ws = np.array([[0.50, 0.35, 0.15],
               [0.35, 0.45, 0.20],
               [0.25, 0.35, 0.40]])


for s1 in range(27):
	for s2 in range(27):
		prob_ij = 1
		for i in range(3):
			prob_ij = prob_ij * ws[int(states[s1, i]) - 1, int(states[s2, i]) - 1]
		prob_ws[s1, s2] = prob_ij

print('prob_ws')
print(prob_ws)

prob_wx = np.ones((27, 27))

wx = np.array([[0.6, 0.3, 0.1],
               [0.2, 0.6, 0.2],
               [0.1, 0.3, 0.6]])

for x in range(27):
	for s in range(27):
		prob_jk = 1
		for i in range(3):
			prob_jk = prob_jk * wx[int(states[x, i]) - 1, int(states[s, i]) - 1]
		prob_wx[x, s] = prob_jk

print('prob_wx')
print(prob_wx)


a_ijk = np.ones((27, 27, 27))

for k in range(27):
	for s1 in range(27):
		for s2 in range(27):
			a_ijk[k, s1, s2] = prob_ws[s1, s2] * prob_wx[k, s2]

# PLOT RESULTS #
fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig1.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')

xpos = []
ypos = []
zpos = []
dz1 = []
dz2 = []

for s1 in range(1, 27):
    for s2 in range(1, 27):
        xpos.append(s1)
        ypos.append(s2)
        zpos.append(0)
        dz1.append(prob_ws[s1, s2])
        dz2.append(prob_wx[s1, s2])
	    
num_elements = len(xpos)
dx = np.ones(1)
dy = np.ones(1)

colors1 = plt.cm.jet((np.asanyarray(dz1).flatten()) / (float(np.asanyarray(dz1).max())))
colors2 = plt.cm.jet((np.asanyarray(dz2).flatten()) / (float(np.asanyarray(dz2).max())))

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz1, color=colors1)
ax2.bar3d(xpos, ypos, zpos, dx, dy, dz2, color=colors2)

plt.show()
