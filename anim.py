import numpy as np
import matplotlib.pyplot as plt
from numeth.ising import ising, set_seed
from time import perf_counter
from matplotlib.animation import FuncAnimation
from time import time

t = time()
set_seed( int((t- int(t))*1000) )

def beta(i):
    return 2*np.exp(i/100)

S = ising(N=256, beta=beta(0), J=0.1, h=0.0, N_iter=0)
fig, ax = plt.subplots()
ax.axis("off")
img = ax.matshow(S, cmap='gray')
title = fig.suptitle("uuu")

def update(i):
    print(f"beta = {beta(i)} (i = {i})")
    ising(N=256, beta=beta(i), J=0.1, h=0.0, N_iter=10, startfrom=S)
    img.set_data(S)
    M = np.mean(S)
    title.set_text(f"beta = {beta(i):.1f} M = {M:.1f}")

anim = FuncAnimation(fig, update, interval=1000/60,frames=800)
# anim.save("anim.mp4")
plt.show()