import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from sys import maxint
from scipy.optimize import lsq_linear
import DBN_Exact_Inference as Inference


### read training data ##################################################################################################################################################################
training_data = Inference.training_data
# print training_data
### initial parameters ##################################################################################################################################################################
t_parameters = {'wtb': -0.004, 'wtp': 0.1, 'wtd': 0.1, 'sigma': 0.01}
i_parameters = {'wib': -.0, 'wit': .0, 'wid': .0, 'wie': .0}
f_parameters = {'beta_c': 0.1, 'kappa_c': 0.1, 'o_c': 0.1}
c_parameters = {'sigma': 0.01}
rand.seed(6)
t0 = rand.uniform(0, 1)
trust = [t0]
delta_trust = []
training_len = len(training_data)
mu_t = [0.5]
A_matrix = []
### trust parameters learning ###########################################################################################################################################################
### The maximization method is based on lsq_linear, min(0.5 * || Ax - B ||**2), s.t. lb <= x <= ub
### A matrix is [1, p, dp], B_matrix is [dt]
for i in range(0, training_len):
	trust.append(rand.normalvariate(mu_t[-1], t_parameters['sigma']))
	delta_trust.append(trust[-1] - trust[-2])
	if i == 0:
		mu_t.append(trust[-1] + t_parameters['wtb'] + t_parameters['wtp'] * training_data.at[i, 'performance'])
	else:
		mu_t.append(trust[-1] + t_parameters['wtb'] + t_parameters['wtp'] * training_data.at[i, 'performance'] + t_parameters['wtd'] * (training_data.at[i, 'performance'] -
                                                                                                                               training_data.at[i-1, 'performance']))
B_matrix = delta_trust
# print 'trust:', trust, '\n', 'delta trust vector:', delta_trust, '\n', 'mu_t:', mu_t

for k in range(0, training_len):
	if k == 0:
		A_matrix.append([1, training_data.at[k, 'performance'], training_data.at[k, 'performance']])
	else:
		A_matrix.append([1, training_data.at[k, 'performance'], training_data.at[k, 'performance'] - training_data.at[k-1, 'performance'] ])
	
# print 'A matrix', A_matrix
lb = np.ones(3) * -10. ** 8
ub = np.ones(3) * 10. ** 8
sigma_new = abs(np.var(trust))
# print 'variance of tk:', sigma_new
t_parameters_learned = lsq_linear(A_matrix, B_matrix, bounds = (lb, ub), lsmr_tol = 'auto', verbose = 1)
t_parameters['wtb_new'] = t_parameters_learned.x[0]
t_parameters['wtp_new'] = t_parameters_learned.x[1]
t_parameters['wtd_new'] = t_parameters_learned.x[2]
# print t_parameters_learned
print t_parameters_learned.x
print



### intervention parameters learning ####################################################################################################################################################
### Matrices are X, S with vectors w_vector, i_vector, mu_vector
### IRLS algorithm for model fitting of Logistic Regression --> w_k+1 = (Xtr x S_k x X)^-1 x Xtr x (S_k x X x w_vector + y_vector - mu_vector
### Use updated Wtb, Wtp, Wtd weights to sample new trusts for the length training data
trust_new = [t0]
d_trust_new = []
mu0 = t0 + t_parameters_learned.x[0] + t_parameters_learned.x[1] * training_data.at[0, 'performance'] + t_parameters_learned.x[2] * (training_data.at[1, 'performance'] -
                                                                                                                           training_data.at[0, 'performance'])
print 'mu new', mu0
mu_t_new = [mu0]
for i in range(0, training_len):
	t_rand = rand.normalvariate(mu_t_new[-1], sigma_new)
	if t_rand > 1.:
		trust_new.append(1.)
	elif t_rand < 0.:
		trust_new.append(0.)
	else:
		trust_new.append(t_rand)
	d_trust_new.append(trust_new[-1] - trust_new[-2])
	if i == 0:
		mu_t_new.append(trust_new[-1] + t_parameters['wtb_new'] + t_parameters['wtp_new'] * training_data.at[i, 'performance'])
	else:
		mu_t_new.append(trust_new[-1] + t_parameters['wtb_new'] + t_parameters['wtp_new'] * training_data.at[i, 'performance'] + t_parameters['wtd_new'] * (training_data.at[i,
		                                                                                                                                                               'performance'] -
	                                                                                                                                training_data.at[i-1,
                                                                                                                                                                     'performance']))
print 'mu_t_ new:', mu_t_new
plt.plot(trust_new)
plt.show()

trust_new.pop(0)
e_vector = training_data['exter'].values
mu_i = []
mu_s = []
w_vector = np.array([i_parameters['wib'], i_parameters['wit'], i_parameters['wid'], i_parameters['wie']])
i_vector = []
y_vector = []
X = np.array([[1 for i in range(training_len)], trust_new, d_trust_new, e_vector])
wx = 0
# w_update = w_vector

###
intervene = np.array(training_data['intervention'].values)
print 'intervene', intervene
w_update = np.array([0., 0., 0., 0.])
w0 = np.log(np.mean(intervene) / (1 - np.mean(intervene)))
X = X.transpose()

for itter in range(100):
	mu_k = []
	eta_i = []
	s_i = []
	z_i = []
	for k in range(0, training_len):
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
		z_i.append(eta_i[-1] + (intervene[k] - mu_k[-1]) / s_i[-1])
	S = np.diagflat(s_i)
	# print "S:", S, 'shape:', S.shape, 'eta', eta
	w_update = (((np.linalg.inv((X.transpose().dot(S)).dot(X))).dot(X.transpose())).dot(S)).dot(z_i)
	print 'w_updated', w_update


