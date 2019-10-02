# -*- coding:   utf-8 -*-
# @author: Maziar Fooladi Mahani

# This code calculates online filtering inference for Multi-Agent Systems

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm, logistic, rv_discrete

# Performances (p1, p2) and Intervention (I) and Levels of Autonomy (L1, L2)
# np.random.seed(11)
# K = 100
# p1 = np.arange(0., 1., 1. / K)
# p2 = np.arange(0., 1., 1. / K)
# I = np.concatenate((np.zeros(K - 50), np.ones(10), np.ones(10) * 2., np.zeros(10), np.ones(10), np.ones(10) * 2.))
# L1 = np.zeros(K)
# L2 = np.ones(K) / 2.


class Inference:
    trustWeights = np.array([.2, .7, .3])
    sigma_t = 0.35
    bin = 20.
    wt1 = [1., -5., 2.5]
    wd1 = [1., -5., .5]
    wt2 = [1., 2.5, -5.]
    wd2 = [1., .5, -5.]
    wb = [5., .1, .1]
    sigma_l = 0.4

    def __init__(self):
        self.vec_t = np.arange(0., 1. + 1. / self.bin, 1. / self.bin)
        self.del_t = np.subtract.outer(self.vec_t, self.vec_t).T
        self.trustBelInit = np.multiply(np.ones(len(self.vec_t)), 0.001)
        self.t1Bel = [self.trustBelInit]
        self.t2Bel = [self.trustBelInit]

    def Prob_t(self, p):
        self.pt = norm.pdf(self.del_t, np.sum(np.multiply(np.array([1., p[1], p[1] - p[0]]), self.trustWeights)), self.sigma_t)

        return self.pt


    def Prob_I(self, I):
        wT1i0xT1 = np.multiply(self.vec_t, self.wt1[0])
        wD1i0DT1 = np.multiply(self.del_t, self.wd1[0])
        wTDT1i0 = wT1i0xT1 + wD1i0DT1
        expT1i0 = np.exp(wTDT1i0)
        wT2i0xT2 = np.multiply(self.vec_t, self.wt2[0])
        wD2i0DT2 = np.multiply(self.del_t, self.wd2[0])
        wTDT2i0 = wT2i0xT2 + wD2i0DT2
        expT2i0 = np.exp(wTDT2i0)
        expT1T2i0 = np.kron(expT2i0, expT1i0)
        expT2T1i0 = np.kron(expT1i0, expT2i0)
        Pi0_num1 = np.multiply(expT1T2i0, np.exp(self.wb[0]))
        Pi0_num2 = np.multiply(expT2T1i0, np.exp(self.wb[0]))

        w1t1i1 = np.multiply(self.vec_t, self.wt1[1])
        w1d1i1 = np.multiply(self.del_t, self.wd1[1])
        t1i1 = w1t1i1 + w1d1i1
        expT1i1 = np.exp(t1i1)
        w2t2i1 = np.multiply(self.vec_t, self.wt2[1])
        w2d2i1 = np.multiply(self.del_t, self.wd2[1])
        t2i1 = w2t2i1 + w2d2i1
        expT2i1 = np.exp(t2i1)
        expT1T2i1 = np.kron(expT2i1, expT1i1)
        expT2T1i1 = np.kron(expT1i1, expT2i1)
        Pi1_num1 = np.multiply(expT1T2i1, np.exp(self.wb[1]))
        Pi1_num2 = np.multiply(expT2T1i1, np.exp(self.wb[1]))

        w1t1i2 = np.multiply(self.vec_t, self.wt1[2])
        w1d1i2 = np.multiply(self.del_t, self.wd1[2])
        t1i2 = w1t1i2 + w1d1i2
        expT1i2 = np.exp(t1i2)
        w2t2i2 = np.multiply(self.vec_t, self.wt2[2])
        w2d2i2 = np.multiply(self.del_t, self.wd2[2])
        t2i2 = w2t2i2 + w2d2i2
        expT2i2 = np.exp(t2i2)
        expT1T2i2 = np.kron(expT2i2, expT1i2)
        expT2T1i2 = np.kron(expT1i2, expT2i2)
        Pi2_num1 = np.multiply(expT1T2i2, np.exp(self.wb[2]))
        Pi2_num2 = np.multiply(expT2T1i2, np.exp(self.wb[2]))

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


    def bel(self, p1, p2, I, L1, L2):

        self.sim_pt1 = self.Prob_t(p1)
        self.sim_pt2 = self.Prob_t(p2)

        self.sim_pi1, self.sim_pi2 = self.Prob_I(I)

        if I == 1:
            self.sim_pl1 = norm.pdf(L1, self.vec_t, self.sigma_l)
            self.sim_pl2 = np.ones(len(self.vec_t)) * 1.
        elif I == 2:
            self.sim_pl1 = np.ones(len(self.vec_t)) * 1.
            self.sim_pl2 = norm.pdf(L2, self.vec_t, self.sigma_l)
        elif I == 0:
            self.sim_pl1 = np.ones(len(self.vec_t)) * 1.
            self.sim_pl2 = np.ones(len(self.vec_t)) * 1.

        self.sim_Pi1_4d = self.sim_pi1.reshape((len(self.vec_t), len(self.vec_t), len(self.vec_t), len(self.vec_t)))
        self.sim_pT2I = np.multiply(self.sim_Pi1_4d, self.sim_pt2)
        self.sim_sumT2_pT2I = np.sum(self.sim_pT2I, axis=3)
        self.sim_sumT2_PT2IBelT2 = np.sum(np.multiply(self.sim_sumT2_pT2I, self.t2Bel[-1]), axis=2)
        self.sim_sumT2_PT2IBelT2xpT1 = np.multiply(self.sim_sumT2_PT2IBelT2, self.sim_pt1)
        self.sim_sumT2_PT2IBelT2xpT1xBelT1xpL1 = np.multiply(np.multiply(self.sim_sumT2_PT2IBelT2xpT1, self.t1Bel[-1]), self.sim_pl1)
        self.Sim_BelT1_numerator = np.sum(self.sim_sumT2_PT2IBelT2xpT1xBelT1xpL1, axis=0)
        self.Sim_BelT1_denumerator = np.sum(self.Sim_BelT1_numerator)
        self.Sim_BelT1 = np.divide(self.Sim_BelT1_numerator, self.Sim_BelT1_denumerator)
        self.t1Bel.append(self.Sim_BelT1)
        print(self.t1Bel)

        self.sim_Pi2_4d = self.sim_pi2.reshape((len(self.vec_t), len(self.vec_t), len(self.vec_t), len(self.vec_t)))
        self.sim_pT1I = np.multiply(self.sim_Pi2_4d, self.sim_pt1)
        self.sim_sumT1_pT1I = np.sum(self.sim_pT1I, axis=3)
        self.sim_sumT1_PT1IBelT1 = np.sum(np.multiply(self.sim_sumT1_pT1I, self.t1Bel[-2]), axis=2)
        self.sim_sumT1_PT1IBelT1xpT1 = np.multiply(self.sim_sumT1_PT1IBelT1, self.sim_pt2)
        self.sim_sumT1_PT1IBelT1xpT1xBelT2xpL2 = np.multiply(np.multiply(self.sim_sumT1_PT1IBelT1xpT1, self.t2Bel[-1]), self.sim_pl2)
        self.Sim_BelT2_numerator = np.sum(self.sim_sumT1_PT1IBelT1xpT1xBelT2xpL2, axis=0)
        self.Sim_BelT2_denumerator = np.sum(self.Sim_BelT2_numerator)
        self.Sim_BelT2 = np.divide(self.Sim_BelT2_numerator, self.Sim_BelT2_denumerator)
        self.t2Bel.append(self.Sim_BelT2)
        print(self.t2Bel)

        return self.t1Bel, self.t2Bel


if __name__=="__main__":

    agent1Inference = Inference()

    agent1Inference.bel([1.,1.], [.5, .5], 1, 0.5, 0.5)
    agent1Inference.bel([1.,1.], [.5, .5], 1, 0.5, 0.5)

