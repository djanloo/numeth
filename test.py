from numeth.core import harmosc

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from numeth.ising import ising, set_seed
from time import perf_counter
from matplotlib.animation import FuncAnimation
from time import time

t = time()
set_seed( int((t- int(t))*1000) )

def beta(i):
    return 10*np.exp(i/100)

S = ising(N=400, beta=beta(0), J=0.1, h=0.0, N_iter=0)
fig, ax = plt.subplots()
img = ax.imshow(S)

def update(i):
    print(f"beta = {beta(i)}")
    ising(N=400, beta=beta(i), J=0.1, h=0.0, N_iter=10, startfrom=S)
    img.set_data(S)

anim = FuncAnimation(fig, update, interval=0,frames=500)
# anim.save("anim.mp4")
plt.show()