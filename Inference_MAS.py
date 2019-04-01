#!python
# !/14usr/Bin/env python
# from scipy.io import loadmat
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm, logistic, rv_discrete


np.random.seed(11)
K = 400

p1 = np.arange(0., 1., 1. / 400.)
p2 = np.arange(0., 1., 1. / 400.)
# p1 = np.concatenate((np.ones(30) * .6, np.ones(30) * .1, np.ones(40) * .1))
# p2 = np.concatenate((np.ones(30) * .6, np.ones(30) * .9, np.ones(40) * .6))
# p1 = np.random.random((K)) ** .1
# p2 = np.random.random((K))

print 'p1:', p1
print 'p2:', p2
# I = np.random.random_integers(0, 2, K)
I = np.concatenate((np.zeros(K - 300), np.ones(50), np.ones(50) * 2., np.zeros(100), np.ones(50), np.ones(50) * 2.))
L1 = np.zeros(400)
L2 = np.ones(400) / 2.
# I = np.zeros(K) * 2.
# L1 = np.random.random_integers(0, 1, K) / 2.
# L2 = np.random.random_integers(0, 1, K) / 2.
print 'I:', I
print 'L1:', L1
print 'L2:', L2

### Prob(Tk|Tk-1, Pk, Pk-1) ####################################################
wt = np.array([.2, .7, .3])
sigma_t = 0.15
Bin = 50.
vec_t = np.arange(0., 1. + 1. / Bin, 1. / Bin)
del_t = np.subtract.outer(vec_t, vec_t).T

def Prob_t(p0, p1):
	pt = norm.pdf(del_t, np.sum(np.multiply(np.array([1., p1, p1 - p0]), wt)), sigma_t)
	return pt

beta = np.array([-2., 4., 6., 3.])

def Prob_t_logit(p0, p1):
	beta_p = np.delete(beta, 1)
	betaxp = np.sum(np.multiply(beta_p, np.array([1, p1, p1 - p0])))
	t0betaxp = vec_t * beta[1] + betaxp
	t0betaxpxt1 = np.multiply.outer(t0betaxp, vec_t).T
	denum = 1. + np.exp(t0betaxp)
	pt = np.transpose(np.divide(np.transpose(np.exp(t0betaxpxt1)), denum))
	return pt

### Prob(Ik|T1k, T1k-1, T2k, T2k-1) #############################################
wt1 = [4., -5., 5.]
wd1 = [4., -2., 2.]
wt2 = [4., 5., -5.]
wd2 = [4., 2., -2.]
wb = [0.1, 2., 2.]

def Prob_I(I):
	wT1i0xT1 = np.multiply(vec_t, wt1[0])
	wD1i0DT1 = np.multiply(del_t, wd1[0])
	wTDT1i0 = wT1i0xT1 + wD1i0DT1
	expT1i0 = np.exp(wTDT1i0)
	wT2i0xT2 = np.multiply(vec_t, wt2[0])
	wD2i0DT2 = np.multiply(del_t, wd2[0])
	wTDT2i0 = wT2i0xT2 + wD2i0DT2
	expT2i0 = np.exp(wTDT2i0)
	expT1T2i0 = np.kron(expT2i0, expT1i0)
	expT2T1i0 = np.kron(expT1i0, expT2i0)
	Pi0_num1 = np.multiply(expT1T2i0, np.exp(wb[0]))
	Pi0_num2 = np.multiply(expT2T1i0, np.exp(wb[0]))
	
	w1t1i1 = np.multiply(vec_t, wt1[1])
	w1d1i1 = np.multiply(del_t, wd1[1])
	t1i1 = w1t1i1 + w1d1i1
	expT1i1 = np.exp(t1i1)
	w2t2i1 = np.multiply(vec_t, wt2[1])
	w2d2i1 = np.multiply(del_t, wd2[1])
	t2i1 = w2t2i1 + w2d2i1
	expT2i1 = np.exp(t2i1)
	expT1T2i1 = np.kron(expT2i1, expT1i1)
	expT2T1i1 = np.kron(expT1i1, expT2i1)
	Pi1_num1 = np.multiply(expT1T2i1, np.exp(wb[1]))
	Pi1_num2 = np.multiply(expT2T1i1, np.exp(wb[1]))
	
	w1t1i2 = np.multiply(vec_t, wt1[2])
	w1d1i2 = np.multiply(del_t, wd1[2])
	t1i2 = w1t1i2 + w1d1i2
	expT1i2 = np.exp(t1i2)
	w2t2i2 = np.multiply(vec_t, wt2[2])
	w2d2i2 = np.multiply(del_t, wd2[2])
	t2i2 = w2t2i2 + w2d2i2
	expT2i2 = np.exp(t2i2)
	expT1T2i2 = np.kron(expT2i2, expT1i2)
	expT2T1i2 = np.kron(expT1i2, expT2i2)
	Pi2_num1 = np.multiply(expT1T2i2, np.exp(wb[2]))
	Pi2_num2 = np.multiply(expT2T1i2, np.exp(wb[2]))

	sum_Pis1 = Pi0_num1 + Pi1_num1 + Pi2_num1
	sum_Pis2 = Pi0_num2 + Pi1_num2 + Pi2_num2
	
	if I == 0:
		Pi1 = np.divide(Pi0_num1, sum_Pis1)
		Pi2 = np.divide(Pi0_num2, sum_Pis2)
	elif I == 1:
		Pi1 = np.divide(Pi1_num1, sum_Pis1)
		Pi2 = np.divide(Pi1_num2, sum_Pis2)
	elif I == 2:
		Pi1 = np.divide(Pi2_num1, sum_Pis1)
		Pi2 = np.divide(Pi2_num2, sum_Pis2)
	
	return Pi1, Pi2


sigma_l = 0.1

### initial values for filtering Belief ###########################################
belT1 = np.multiply(np.ones(len(vec_t)), 0.001)
belT2 = np.multiply(np.ones(len(vec_t)), 0.001)
Bel_T1 = [belT1]
Bel_T2 = [belT2]


for k in range(K-1):
	sim_pt1 = Prob_t(p1[k], p1[k + 1])
	sim_pt2 = Prob_t(p2[k], p2[k + 1])
	# sim_pt1 = Prob_t_logit(p1[k], p1[k + 1])
	# sim_pt2 = Prob_t_logit(p2[k], p2[k + 1])
	[sim_pi1, sim_pi2] = Prob_I(I[k])
	
	if I[k] == 1:
		sim_pl1 = norm.pdf(L1[k], vec_t, sigma_l)
		sim_pl2 = np.ones(len(vec_t)) * 1.
	elif I[k] == 2:
		sim_pl1 = np.ones(len(vec_t)) * 1.
		sim_pl2 = norm.pdf(L2[k], vec_t, sigma_l)
	elif I[k] == 0:
		sim_pl1 = np.ones(len(vec_t)) * 1.
		sim_pl2 = np.ones(len(vec_t)) * 1.
	
	sim_Pi1_4d = sim_pi1.reshape((len(vec_t), len(vec_t), len(vec_t), len(vec_t)))
	sim_pT2I = np.multiply(sim_Pi1_4d, sim_pt2)
	sim_sumT2_pT2I = np.sum(sim_pT2I, axis = 3)
	sim_sumT2_PT2IBelT2 = np.sum(np.multiply(sim_sumT2_pT2I, Bel_T2[-1]), axis = 2)
	sim_sumT2_PT2IBelT2xpT1 = np.multiply(sim_sumT2_PT2IBelT2, sim_pt1)
	sim_sumT2_PT2IBelT2xpT1xBelT1xpL1 = np.multiply(np.multiply(sim_sumT2_PT2IBelT2xpT1, Bel_T1[-1]), sim_pl1)
	Sim_BelT1_numerator = np.sum(sim_sumT2_PT2IBelT2xpT1xBelT1xpL1, axis = 0)
	Sim_BelT1_denumerator = np.sum(Sim_BelT1_numerator)
	Sim_BelT1 = np.divide(Sim_BelT1_numerator, Sim_BelT1_denumerator)
	Bel_T1.append(Sim_BelT1)
	
	sim_Pi2_4d = sim_pi2.reshape((len(vec_t), len(vec_t), len(vec_t), len(vec_t)))
	sim_pT1I = np.multiply(sim_Pi2_4d, sim_pt1)
	sim_sumT1_pT1I = np.sum(sim_pT1I, axis = 3)
	sim_sumT1_PT1IBelT1 = np.sum(np.multiply(sim_sumT1_pT1I, Bel_T1[-2]), axis = 2)
	sim_sumT1_PT1IBelT1xpT1 = np.multiply(sim_sumT1_PT1IBelT1, sim_pt2)
	sim_sumT1_PT1IBelT1xpT1xBelT2xpL2 = np.multiply(np.multiply(sim_sumT1_PT1IBelT1xpT1, Bel_T2[-1]), sim_pl2)
	Sim_BelT2_numerator = np.sum(sim_sumT1_PT1IBelT1xpT1xBelT2xpL2, axis = 0)
	Sim_BelT2_denumerator = np.sum(Sim_BelT2_numerator)
	Sim_BelT2 = np.divide(Sim_BelT2_numerator, Sim_BelT2_denumerator)
	Bel_T2.append(Sim_BelT2)
	
var_BelT1 = np.std(Bel_T1, axis = 1)
var_BelT2 = np.std(Bel_T2, axis = 1)


sharp_bel = np.ones(len(vec_t)) / Bin
# sharp_bel[10]  = 1.

MDS1 = []
for k in range(K):
	MD = 0.
	for i in range(len(vec_t)):
		for  j in range(len(vec_t)):
			MD += Bel_T1[k][i] * Bel_T1[k-1][j] * abs(np.float(i) / len(vec_t) - np.float(j) / len(vec_t))
	MD = MD / Bin ** 2
	MDS1.append(MD)
# print MDS1

MDS2 = []
for k in range(K):
	MD = 0.
	for i in range(len(vec_t)):
		for  j in range(len(vec_t)):
			MD += Bel_T2[k][i] * Bel_T2[k-1][j] * abs(np.float(i) / len(vec_t) - np.float(j) / len(vec_t))
	MD = MD / Bin ** 2
	MDS2.append(MD)
# print MDS2
print

### plot the data ############################################################
X, Y = np.meshgrid(np.arange(0, K, 1), vec_t)
Z1 = np.asarray(map(list, zip(*Bel_T1)))
Z2 = np.asarray(map(list, zip(*Bel_T2)))

max_bel = 0.
min_bel = 0.
for i in range(len(Z1)):
	if max_bel < max(Z1[i]):
		max_bel = max(Z1[i])
	if min_bel > min(Z1[i]):
		min_bel = min(Z1[i])
data1 = Z1
data2 = Z2
ypos1, xpos1 = np.indices(data1.shape)
ypos2, xpos2 = np.indices(data2.shape)
xpos1 = xpos1.flatten()
xpos2 = xpos2.flatten()
ypos1 = ypos1.flatten()
ypos2 = ypos2.flatten()
zpos1 = np.zeros(xpos1.shape)
zpos2 = np.zeros(xpos2.shape)

fig1 = plt.figure(1)
ax = Axes3D(fig1)
ax.view_init(elev = 90., azim = -90.)
colors = plt.cm.jet(data1.flatten() / float(data1.max()))
ax.bar3d(xpos1, ypos1, zpos1, .8, .8, data1.flatten(), color = colors)
ax.set_xlabel('Time')
ax.set_ylabel('Trust')

fig2 = plt.figure(2)
ax2 = Axes3D(fig2)
ax2.view_init(elev = 90., azim = -90.)
colors = plt.cm.jet(data2.flatten() / float(data2.max()))
ax2.bar3d(xpos2, ypos2, zpos2, .8, .8, data2.flatten(), color = colors)
ax2.set_xlabel('Time')
ax2.set_ylabel('Trust')

fig3, (axp1, axp2, axvar1, axvar2) = plt.subplots(4, 1)
axp1.plot(np.arange(K), p1)
axp2.plot(np.arange(K), p2)
axvar1.plot(np.arange(K), MDS1)
axvar2.plot(np.arange(K), MDS2)
plt.show()
