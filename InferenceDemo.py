# -*- coding:   utf-8 -*-
# @author: Maziar Fooladi Mahani

import tkinter as tk
from tkinter import  ttk
import Inference_Realtime
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
import matplotlib.pyplot as plt
from random import randint
import time
from matplotlib.figure import Figure
import pygame
import numpy as np

pygame.init()

pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)

joystick.init()





master = tk.Tk()
master.geometry("1000x1000+900+50")

per_frame = tk.Frame(master)
per_frame.pack(side=tk.LEFT, padx = 30, )

per_1 = tk.Scale(per_frame, from_ = 1, to = 0, resolution = 0.01, length = 350, width = 35, cursor = 'spider', takefocus = True, label = 'Per. 1', troughcolor = '#000000')
per_1.set(.5)
per_1_pre = per_1.get()
per_1.pack(pady = 50)

per_2 = tk.Scale(per_frame, from_ = 1, to = 0, resolution = 0.01, length = 350, width = 35, cursor = 'spider', takefocus = True, label = 'Per. 2', troughcolor = '#000000')
per_2.set(.5)
per_2_pre = per_2.get()
per_2.pack(pady  = 50)

loa_frame = tk.Frame(master)
loa_frame.pack(side = tk.LEFT, padx = 30)

loa_1 = tk.Scale(loa_frame, from_ = 1, to = 0, resolution = 0.5, length = 350, width = 35, cursor = 'trek', takefocus = True, label = 'LoA 1', troughcolor = '#000000')
loa_1.set(1)
loa_1_pre = 1.
loa_1.pack(pady = 50)

loa_2 = tk.Scale(loa_frame, from_ = 1, to = 0, resolution = 0.5, length = 350, width = 35, cursor = 'trek', takefocus = True, label = 'LoA 2', troughcolor = '#000000')
loa_2.set(1)
loa_2_pre = 1.
loa_2.pack(pady  = 50)


time_frame = tk.Frame(master)
time_frame.pack(side = tk.LEFT, padx = 30)
start = time.perf_counter()

time_label = tk.Label(master, text = time.perf_counter() - start)
time_label.pack()



fig_frame = tk.Frame(master)
fig_frame.pack()

f = Figure(figsize=(20, 20), dpi=100)
a = f.add_subplot(211)
a2 = f.add_subplot(212)
canvas = FigureCanvasTkAgg(f, fig_frame)
canvas._tkcanvas.pack(side=tk.RIGHT)
limits = 10

def init():
    a.set_xlim(0, 1)
    a.set_ylim(0, 1.1)
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1.1)
    return a, a2

time_line = []
performance1 = []
performance2 = []

level_of_autonomy1 = []
level_of_autonomy2 = []

time_line_data = []
trust_vec = np.arange(0,1,1./21.)
trust_data = []
def update(frame):
    global loa_2_pre, loa_1_pre, per_1_pre, per_2_pre, time_line_data, trust_data
    # only one LoA could be not 1 so if one is already less than one and the user has changed the other one the first one is set back to 1
    if loa_1.get() != 1 and loa_2.get() != 1:
        if (loa_1.get() - loa_1_pre != 0):
            loa_2.set(1)
        elif (loa_2.get() - loa_2_pre != 0):
            loa_1.set(1)

    if loa_1.get() == 1 and loa_2.get() == 1:
        I = 0
    elif loa_1.get() != 1:
        I = 1
    elif loa_2.get() != 2:
        I = 2

    loa_1_pre = loa_1.get()
    loa_2_pre = loa_2.get()

    bel1, bel2 = inference.bel([per_1.get(), per_1_pre], [per_2.get(), per_2_pre], I, loa_1.get(), loa_2.get())

    # update previous performances
    per_1_pre = per_1.get()
    per_2_pre = per_2.get()

    process_time = time.perf_counter() - start

    a.clear()
    a2.clear()

    a.set_xlim(0, process_time)
    a.set_ylim(0, 1.1)
    a2.set_xlim(0, process_time)
    a2.set_ylim(0, 1.1)

    time_line.append(process_time)

    performance1.append(per_1.get())
    performance2.append(per_2.get())

    level_of_autonomy1.append(loa_1.get())
    level_of_autonomy2.append(loa_2.get())

    time_line_data.append(np.multiply(np.ones(len(bel1[-1])), time_line[-1]))
    trust_data.append(trust_vec)

    a.plot(time_line, performance1)
    a.plot(time_line, level_of_autonomy1, 'r')
    a.scatter(time_line_data, trust_data, c=bel1[1:], cmap = 'YlOrRd')
    a2.plot(time_line, performance2)
    a2.plot(time_line, level_of_autonomy2, 'r')
    a2.scatter(time_line_data, trust_data, c=bel2[1:], cmap = 'YlOrRd')

    canvas.draw()
    time_label.config(text = process_time)

    return a, a2




if __name__ == "__main__":

    inference = Inference_Realtime.Inference()
    # ag2 = Inference_Realtime.Inference()



    ani = FuncAnimation(f, update, init_func=init)

    tk.mainloop()

