
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randint


fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    return ln,

def update(frame):
    xdata = [randint(0, 20), randint(0, 20), randint(0, 20), randint(0, 20), randint(0, 20)]
    ydata = [randint(0, 20), randint(0, 20), randint(0, 20), randint(0, 20), randint(0, 20)]
    ln.set_data(xdata, ydata)

    return ln,

ani = FuncAnimation(fig, update,
                    init_func=init)
plt.show()