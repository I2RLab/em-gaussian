#!python
#!/usr/Bin/env python
from scipy.io import loadmat
x = loadmat('data_sample.mat')
f_k = x['f_k'][0]
i_k = x['i_k'][0]
e_k = x['e_k'][0]
p_k = x['p_k'][0]
c_k = x['c_k'][0]

print len(f_k), len(i_k), len(e_k), len(p_k), len(c_k)