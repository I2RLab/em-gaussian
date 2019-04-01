import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from sys import maxint
from scipy.optimize import lsq_linear
import DBN_Exact_Inference as Inference
rand.seed(6)
x1 = []
x2 = []
x0 = np.ones(100)
for i in range(100):
	x1.append(rand.normalvariate(0.5, .1))
	x2.append(rand.normalvariate(0.5, .2))
# print 'x1:', x1, '\n', 'x2:', x2, '\n', 'x0:', x0

ws = np.array([-1., 1., 1.])
wx = ws.dot([x0, x1, x2])
y = 1. / (1. + np.exp(-wx))
print 'ws', ws, '\n', 'wx', wx, '\n', 'y', y

intervene = []

for i, yk in enumerate(y):
	if yk < 0.5:
		intervene.append(0.)
	else:
		intervene.append(1.)

w_update = np.array([1., 1., 1.])
eta_k = []
eta = []
X = np.array([x0, x1, x2]).transpose()
print 'X:', X

for itter in range(1000):
	for k in range(0, len(x1)):
		mu = w_update.dot(X[k].transpose())
		if mu > 10.:
			mu = 10.
		elif mu < -10.:
			mu = -10.
		if intervene[k] == 1.:
			eta_k.append((1. + np.exp(-mu))** -1)
		else:
			eta_k.append(1. - 1. / (1. + np.exp(-mu)))
		
		eta.append(eta_k[-1] * (1. - eta_k[-1]))

 	S = np.diagflat(eta)
	# print "S:", S, 'shape:', S.shape, 'eta', eta
	w_update = ((np.linalg.inv((X.transpose().dot(S)).dot(X))).dot(X.transpose())).dot((S.dot(X)).dot(w_update.transpose()) + np.array(intervene) - np.array(eta_k))
	print 'w_updated', w_update
	eta_k = []
	eta = []
	# S = []
	
# plt.plot(x1,'*')
# plt.plot(x2,'.')
plt.plot(wx, y, '*')
plt.plot(wx, intervene, '*')
plt.show()