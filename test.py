import numpy as np
import matplotlib.pyplot as plt
from numeth.ising import ising, set_seed
from matplotlib.animation import FuncAnimation
from time import time
from rich.progress import track
from telegram_send import send

t = time()
set_seed( int((t- int(t))*10000) )

def beta(i):
    return 2*np.exp(i/100)

npoints = 40
n_means = 100

S = ising(N=100, beta=500, J=1.0, h=0.0, N_iter=100)
m = np.zeros(npoints)
T = np.linspace(2.0, 3, npoints)[::-1]

for i, t in track(enumerate(T), total=npoints):
    for _ in range(n_means):
        ising(N=100, beta=1/t, J=1.0, h=0.0005, N_iter=70, startfrom=S)
        m[i] += np.mean(S)
    m[i] /= n_means


plt.plot(T, m)
plt.ylabel("$\psi$")
plt.xlabel("T")

plt.savefig("figure.png")

with open("figure.png", "rb") as f:
    send(images=[f])

plt.show()