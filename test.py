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

npoints = 20
n_means = 30

m = np.zeros(npoints)
sigmas = np.zeros(npoints)
T = np.linspace(0.01, 4.0, npoints)[::-1]

for i, t in track(enumerate(T), total=npoints):
    mm = np.zeros(n_means)
    S = ising(N=100, beta=1/t, J=1.0, h=0.007, N_iter=512 )

    for sample in range(n_means):
        mm[sample] = np.mean(S)
        ising(N=100, beta=1/t, J=1.0, h=0.007, N_iter=128, startfrom=S)
    print(f"t = {t}: mm = {mm}")
    m[i] = np.mean(mm)
    sigmas[i] = np.std(mm)

    # send(messages=[f"{i/npoints*100:.1f}"])

plt.figure(100)
plt.errorbar(T, m, sigmas,  ls="", marker=".")
plt.ylabel("$\psi$")
plt.xlabel("T")

plt.savefig("figure.png")

with open("figure.png", "rb") as f:
    send(images=[f])

plt.show()