from numeth.core import harmosc

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from numeth.ising import ising
from time import perf_counter
from matplotlib.animation import FuncAnimation

S = ising(N=200, beta=5, J=0.1, h=0.0, N_iter=0)
fig, ax = plt.subplots()
img = ax.imshow(S)
def update(i):
    ising(N=200, beta=6.0, J=0.1, h=0.0, N_iter=1, startfrom=S)
    img.set_data(S)
anim = FuncAnimation(fig, update, interval=1000/60,frames=100)
anim.save("anim.mp4")
plt.show()