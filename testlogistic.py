import numpy as np

# Bin = 4.
# vec_t = np.arange(0., 1. + 1. / Bin, 1. / Bin)
# print 'vector t', vec_t
# del_t = np.subtract.outer(vec_t, vec_t).T
# print 'delta t', del_t
#
# wt1 = [1., 2., 3.]  # wt1 = [wt1i0 wt1i1 wt1i2]
# wd1 = [1., 2., 3.]  # wt1 = [wd1i0 wd1i1 wd1i2]
# wt2 = [2., 2., 3.]  # wt1 = [wt2i0 wt2i1 wt2i2]
# wd2 = [2., 2., 3.]  # wt1 = [wd2i0 wd2i1 wd2i2]
# wb = [1., 2., 3.]  # wt1 = [wbi0 wbi1 wbi2]
#
# wT1i0xT1 = np.multiply(vec_t, wt1[0])
# wD1i0DT1 = np.multiply(del_t, wd1[0])
# wTDT1i0 = wT1i0xT1 + wD1i0DT1
# expT1i0 = np.exp(wTDT1i0)
# wT2i0xT2 = np.multiply(vec_t, wt2[0])
# wD2i0DT2 = np.multiply(del_t, wd2[0])
# wTDT2i0 = wT2i0xT2 + wD2i0DT2
# expT2i0 = np.exp(wTDT2i0)
# expT1T2i0 = np.kron(expT1i0, expT2i0)
# Pi0_num1 = np.multiply(expT1T2i0, np.exp(wb[0]))
# expT2T1i0 = np.kron(expT2i0, expT1i0)
# Pi0_num2 = np.multiply(expT2T1i0, np.exp(wb[0]))
# p4d = Pi0_num2.reshape((len(vec_t), len(vec_t), len(vec_t), len(vec_t)))
# print
#
# p = np.ones((3,4,5,6))
# p1 = np.sum(p, axis =0)
a = np.array([1,2,3,4])
b = a * 0.2
print 'b',b , '\n'
bb = np.multiply.outer(b,a).T
print bb
c = np.transpose(np.divide(np.transpose(bb), b))
print c