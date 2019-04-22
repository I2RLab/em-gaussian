import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

x = np.arange(0,10,1)
y = np.random.rand(10)

fig = plt.Figure()
ax = fig.add_subplot(211)

plt.scatter(x,y, c =x)
plt.show()