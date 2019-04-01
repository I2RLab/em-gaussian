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
	x1.append(rand.normalvariate(0.5, .25))
	x2.append(rand.normalvariate(0.5, .25))
# print 'x1:', x1, '\n', 'x2:', x2, '\n', 'x0:', x0

ws = np.array([1.1, -1.3, -1.4])
wx = ws.dot([x0, x1, x2])
y = 1. / (1. + np.exp(-wx))
# print 'ws', ws, '\n', 'wx', wx, '\n', 'y', y

intervene = []

for i, yk in enumerate(y):
	if yk < 0.5:
		intervene.append(0.)
	else:
		intervene.append(1.)

w_update = np.array([0., 0., 0.])
w0 = np.log(np.mean(intervene)/(1 - np.mean(intervene)))
X = np.array([x0, x1, x2]).transpose()

for itter in range(100):
	mu_k = []
	eta_i = []
	s_i = []
	z_i = []
	for k in range(0, len(x1)):
		eta_i.append(w0 + w_update.dot(X[k]))
		
		if eta_i[-1] > 6.:
			eta_i[-1] = 6.
		elif eta_i[-1] < -6.:
			eta_i[-1] = -6.
			
		if intervene[k] == 1.:
			mu_k.append((1. + np.exp(-eta_i[-1])) ** -1)
		else:
			mu_k.append(1. - (1. + np.exp(-eta_i[-1])) ** -1)
		
		s_i.append(mu_k[-1] * (1. - mu_k[-1]))
		z_i.append(eta_i[-1] + (intervene[k] - mu_k[-1])/s_i[-1])
	S = np.diagflat(s_i)
	# print "S:", S, 'shape:', S.shape, 'eta', eta
	w_update = (((np.linalg.inv((X.transpose().dot(S)).dot(X))).dot(X.transpose())).dot(S)).dot(z_i)
	print 'w_updated', w_update
	
