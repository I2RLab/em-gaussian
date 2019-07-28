import numpy as np
c = [12]
a = (np.arange(12)).reshape(4,3)
b = np.ones((3,2)) + [1.,2.]
print('a',a)
print('b',b)
print
matmul_ab = np.matmul(a,b)
print(matmul_ab, matmul_ab.shape, matmul_ab.size)

print(np.multiply(matmul_ab, matmul_ab))
print(matmul_ab + matmul_ab)

