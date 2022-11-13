"""
Esegue la simulazione MC per n_temperature diverse e per ciascuna temperatura media su n_samples campioni indipendenti.

Per ogni temperatura esegue un ciclo di termalizzazione abbassando gradualmente il campo.

Poi per ogni temperatura calcola il parametro d'ordine psi[T, sample]
la statistica campionaria e' quindi calcolata per ogni temperatura.

"""
import numpy as np
import matplotlib.pyplot as plt
from numeth.ising import ising, set_seed, energy
from matplotlib.animation import FuncAnimation
from time import time
from rich.progress import track

def autocorr(y):
    x = y.copy()
    x -= np.mean(x)
    x /= np.std(x)
    result = np.correlate(x, x, mode='full')/len(x)
    return result[result.size//2:]

# Imposta il seed usando il tempo
t = time()
set_seed( int((t- int(t))*10000) )

N_celle = 32
n_temperature = 30
n_samples = 10000

psi = np.zeros((n_temperature, n_samples))
H = np.zeros((n_temperature, n_samples))

mean_psi = np.zeros(n_temperature)
sigma_psi = np.zeros(n_temperature)
mean_H = np.zeros(n_temperature)
sigma_H = np.zeros(n_temperature)

T = np.linspace(-1.0, 1.0, n_temperature)
T = 2.3*(T**3 + 1) + 1e-3

for i, t in track(enumerate(T), total=n_temperature):
    # Field limit + thermalization
    S = ising(N=N_celle, beta=1/t, J=1.031, h=0.0, N_iter=100)
    # for h in [0.003, 0.002, 0.001, 0.0, 0.0, 0.0]:
    #     S = ising(N=N_celle, beta=1/t, J=1.0, h=h, N_iter=1, startfrom=S)

    for sample in range(n_samples):
        psi[i, sample] = np.abs(np.mean(S))
        H[i, sample] = energy(S,1.0,0.0)
        ising(N=N_celle, beta=1/t, J=1.0, h=0.0 , N_iter=50, startfrom=S)

np.save("psi.npy", psi)
np.save("H.npy", H)

# for i,t in enumerate(T):
#     plt.plot(autocorr(psi[i, 50:]), label=f"T={t}")
# plt.legend(fontsize=8)

# plt.figure(2)
# for i,t in enumerate(T):
#     plt.plot(autocorr(H[i, 50:]), label=f"T={t}")
# plt.legend(fontsize=8)

# plt.show()


mean_psi = np.abs(np.mean(psi, axis=1))
sigma_psi = np.std(psi, axis=1)
mean_H = np.mean(H, axis=1)
sigma_H = np.std(H, axis=1)

fig, ax = plt.subplots(2, sharex=True)
ax[0].errorbar(T, mean_psi, sigma_psi, ls="", marker=".")
ax[0].set_xlabel("T")
ax[0].set_ylabel(r"$\langle \psi \rangle$")

ax[1].errorbar(T, mean_H, sigma_H, ls="", marker=".")
ax[1].set_ylabel(r"$\langle H \rangle$")

plt.savefig("psi_H.png")

fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(T, (sigma_psi**2)/T, ls="", marker=".")
ax[0].set_xlabel("T")
ax[0].set_ylabel(r"$\langle \chi \rangle$")

ax[1].plot(T, (sigma_H**2)/T ,  ls="", marker=".")
ax[1].set_ylabel(r"$\langle C_v \rangle$")

plt.savefig("chi_cv.png")
plt.show()
