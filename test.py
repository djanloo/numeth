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
n_means = 50

m = np.zeros(npoints)
sigmas = np.zeros(npoints)
T = np.linspace(0, 3.5, npoints)

for i, t in track(enumerate(T), total=npoints):
    mm = np.zeros(n_means)

    # Field limit + thermalization
    S = ising(N=300, beta=1/t, J=1.031, h=0.1, N_iter=128)
    for h in [0.003, 0.002, 0.001, 0.0, 0.0, 0.0]:
        S = ising(N=300, beta=1/t, J=1.0, h=h, N_iter=100, startfrom=S)

    for sample in range(n_means):
        mm[sample] = np.mean(S)
        ising(N=300, beta=1/t, J=1.0, h=0.0 , N_iter=512, startfrom=S)
    print(f"t = {t}: mm = {mm}")
    m[i] = np.mean(mm)
    sigmas[i] = np.std(mm)


plt.figure(1)
plt.errorbar(T, m, sigmas,  ls="", marker=".")
plt.ylabel("$\psi$")
plt.xlabel("T")

plt.savefig("figure.png")

with open("figure.png", "rb") as f:
    send(images=[f])
