#!python
#!/usr/Bin/env python
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

np.seterr(over = 'ignore')
B = 100.0 # Number of histogram bins
rng = pd.interval_range(0, 1, 10)
trust_vector = pd.Series([i / B for i in range(0,int(B) + 1)])

### OPTIMO paper training data ##########################################################################################################################################################
x = loadmat('data_sample.mat')
f_k = x['f_k'][0]
i_k = x['i_k'][0]
e_k = x['e_k'][0]
p_k = x['p_k'][0]
c_k = x['c_k'][0]

data_index = pd.Series([i for i in range(0, len(p_k))])
data = {'performance':p_k, 'intervention': i_k, 'exter': e_k, 'feedback': f_k, 'change': c_k }

training_data = pd.DataFrame(data = data, index = data_index)

trust_param = {'wtb': 0.0064, 'wtp': 0.0153, 'wtd': 0.0029, 'sigma': 0.05}
intervention_param = {'wib': 131.2, 'wit': -157.1, 'wid': -9887., 'wie': 83.84}
change_param = {'beta_c': 1.063 * 10. ** (-7), 'kappa_c': 1277., 'o_c': 0.0003}
feedback_param = {'sigma': 0.1}


class Trust(object):
	def __init__(self, t_param, i_param, c_param, f_param):
		self.t_param = t_param
		self.i_param = i_param
		self.c_param = c_param
		self.f_param = f_param
		
	def t_prob(self, t0, t1, p0, p1):
		mu = t0 + self.t_param['wtb'] + self.t_param['wtp'] * p1 + self.t_param['wtd'] * (p1 - p0)
		P_t = np.exp((-(t1 - mu) ** 2) / (2. * self.t_param['sigma'] ** 2)) / (self.t_param['sigma'] * np.sqrt(2. * np.pi))
		
		# print 'Pt', P_t
		return P_t
	
	def i_prob(self, t0, t1, i1, e1):
		x = self.i_param['wib'] + self.i_param['wit'] * t1 + self.i_param['wid'] * (t1  - t0) + self.i_param['wie'] * e1
		P_i = 1. / (1. + np.exp(-x))
		
		if i1 == 0.:
			P_i = 1 - P_i
		
		# print 'Pi', P_i
		return P_i
		
	def c_prob(self, t0, t1, c1):
		if c1 == 1:
			P_c = self.c_param['beta_c'] + (1. - 3. * self.c_param['beta_c']) / (1. + np.exp(self.c_param['kappa_c'] * (t1 - t0 - self.c_param['o_c'])))
		elif c1 == -1:
			P_c = self.c_param['beta_c'] + (1. - 3. * self.c_param['beta_c']) / (1. + np.exp(self.c_param['kappa_c'] * (-t1 + t0 - self.c_param['o_c'])))
		elif c1 == 0:
			P_c = 1. - (self.c_param['beta_c'] + (1. - 3. * self.c_param['beta_c']) / (1. + np.exp(self.c_param['kappa_c'] * (t1 - t0 - self.c_param['o_c'])))) - (self.c_param['beta_c']
			                                                                                                                                                    + (1. - 3. *
			                                                                                                                                                       self.c_param[
				                                                                                                                                                       'beta_c']) / (
				                                                                                                                                                      1. + np.exp(self.c_param['kappa_c'] * (-t1 + t0 - self.c_param['o_c']))))
		
		return P_c
	
	def f_prob(self, t1, f1):
		P_f = np.exp(-(f1 - t1) ** 2 / (2. * self.f_param['sigma'] ** 2.)) / (np.sqrt(2. * np.pi) * self.f_param['sigma'])
		
		return P_f
	
	
	def belief_bar(self, t0, t1, p0, p1, i1, e1, c1, f1):
		if (c1 != 2. and f1 != 2.):
			bel_tt = self.t_prob(t0, t1, p0, p1) * self.i_prob(t0, t1, i1, e1) * self.c_prob(t0, t1, c1) * self.f_prob(t1, f1)
		elif (c1 != 2. and f1 == 2.):
			bel_tt = self.t_prob(t0, t1, p0, p1) * self.i_prob(t0, t1, i1, e1) * self.c_prob(t0, t1, c1)
		elif (c1 == 2. and f1 != 2.):
			bel_tt = self.t_prob(t0, t1, p0, p1) * self.i_prob(t0, t1, i1, e1) * self.f_prob(t1, f1)
		elif (c1 == 2. and f1 == 2.):
			bel_tt = self.t_prob(t0, t1, p0, p1) * self.i_prob(t0, t1, i1, e1)
		
		return bel_tt


if __name__ == "__main__":
	BeliefTk = Trust(trust_param, intervention_param, change_param, feedback_param)
	
	belief_tk = []
	belief_t0 = [0.5 for bt in range(0, trust_vector.__len__())]
	belief_tk.append(belief_t0)
	K = len(p_k) # training discrete length
	belief_t0t1 = []
	
	for k in range(1, K):
		beltt = []
		
		for tr1 in range(0, trust_vector.__len__()):
			bel = []
		
			for tr0 in range(0, trust_vector.__len__()):
				bel.append(BeliefTk.belief_bar(trust_vector[tr0], trust_vector[tr1], p_k[k-1], p_k[k], i_k[k], e_k[k], c_k[k], f_k[k]) * belief_tk[k - 1][tr0])
			
			# beltt.append(sum(bel) * 1. / B) # beltt is the belief_filtered for each time step.
			beltt.append(sum(bel)) # beltt is the belief_filtered for each time step.
			belief_t0t1.append(bel)
		
		belief_sum = sum(beltt)
		
		for i in range(0, len(beltt)):
			beltt[i] = beltt[i] / belief_sum
			
		belief_tk.append(beltt) # belief_tf is the belief_filtered history for time=1:K, K training-session discrete steps.
	
	# fig = plt.figure()
	# ax = Axes3D(fig)
	X, Y = np.meshgrid(np.arange(0, K, 1), trust_vector)
	Z = np.asarray(map(list, zip(*belief_tk)))
	max_bel = 0.
	min_bel = 0.
	for i in range(len(Z)):
		if max_bel < max(Z[i]):
			max_bel = max(Z[i])
		if min_bel > min(Z[i]):
			min_bel = min(Z[i])
	# print 'min', min_bel, 'max', max_bel
	
	### Plot the surface
	# surf = ax.plot_surface(X, Y, Z, cmap = cm.coolwarm,
	#                        linewidth = 0, antialiased = False, vmin = 0.0, vmax = max_bel / 15.0)
	#
	# # Customize the z axis.
	# ax.set_zlim(min_bel, max_bel)
	# ax.zaxis.set_major_locator(LinearLocator(10))
	# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	#
	# # Add a color bar which maps values to colors.
	# fig.colorbar(surf, shrink=.5, aspect=5)
	#
	# plt.show()
	### Plot the 3d bars
	data = Z
	ypos, xpos = np.indices(data.shape)
	
	xpos = xpos.flatten()
	ypos = ypos.flatten()
	zpos = np.zeros(xpos.shape)
	
	fig = plt.figure()
	ax = Axes3D(fig)
	# ax = fig.add_subplot(111, projection = '3d')
	# surf = ax.plot_surface(X, Y, Z, cmap = cm.coolwarm,
	#                        linewidth = 0, antialiased = False, vmin = 0.0, vmax = max_bel / 15.0)
	ax.view_init(elev=90., azim=-90.)
	colors = plt.cm.jet(data.flatten() / float(data.max()))
	ax.bar3d(xpos, ypos, zpos, .8, .8, data.flatten(), color = colors)
	# colors.set_array(Z)
	# colBar = fig.colorbar(colors)
	ax.set_xlabel('Time')
	ax.set_ylabel('Trust [0,100]->[0,1]')
	# ax.set_zlabel('Z')
	# ax.zaxis.set_major_locator(LinearLocator(10))
	# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	
	
	
	plt.show()
	print