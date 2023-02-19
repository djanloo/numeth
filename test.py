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
from numeth.utils import autocorr
import seaborn as sns
sns.set()

# Imposta il seed usando il tempo
t = time()
set_seed( int((t- int(t))*10000) )


N_celle = 30
n_temperature = 20
n_samples = 1000

psi = np.zeros((n_temperature, n_samples))
H = np.zeros((n_temperature, n_samples))

mean_psi = np.zeros(n_temperature)
sigma_psi = np.zeros(n_temperature)
mean_H = np.zeros(n_temperature)
sigma_H = np.zeros(n_temperature)

betas = np.linspace(0.35, 0.5, n_temperature)


# psi = np.load(f"psi_L{N_celle}_nT{n_temperature}_nsamp{n_samples}.npy")
# H   = np.load(f"H_L{N_celle}_nT{n_temperature}_nsamp{n_samples}.npy")

for i, beta in track(enumerate(betas), total=n_temperature):
    # print(f"elapsed temperatures: {int(i/len(betas)*100)}%")
    S = ising(N=N_celle, beta=beta, J=1.0, h=0.0, N_iter=2000)

    for sample in range(n_samples):
        psi[i, sample] = np.abs(np.mean(S))
        H[i, sample] = energy(S, 1.0, 0.0)
        ising(N=N_celle, beta=beta, h=0.0 , N_iter=100, startfrom=S)

print("saving")
np.save(f"psi_L{N_celle}_nT{n_temperature}_nsamp{n_samples}.npy", psi)
np.save(f"H_L{N_celle}_nT{n_temperature}_nsamp{n_samples}.npy", H)


######## BOOTSTRAP

samples_of_means_of_m = np.zeros((n_temperature, n_samples))
for t in track(range(n_temperature), description="resampling m.."):
    for samp in range(n_samples):
        indexes = np.random.randint(n_samples , size=n_samples)
        samples_of_means_of_m[t, samp] = np.mean(psi[t, indexes], axis=0)

np.save(f"resampled_mean_m.npy",samples_of_means_of_m)

samples_of_means_of_E = np.zeros((n_temperature, n_samples))
for t in track(range(n_temperature), description="resampling E.."):
    for samp in range(n_samples):
        indexes = np.random.randint(n_samples , size=n_samples)
        samples_of_means_of_E[t, samp] = np.mean(H[t, indexes], axis=0)
np.save(f"resampled_mean_E.npy",samples_of_means_of_E)


# samples_of_means_of_m = np.load("resampled_mean_m.npy")
# samples_of_means_of_E = np.load("resampled_mean_E.npy")/400



mean_psi = np.abs(np.mean(samples_of_means_of_m, axis=1))
sigma_psi = np.std(samples_of_means_of_m, axis=1)
mean_H = np.mean(samples_of_means_of_E, axis=1)
sigma_H = np.std(samples_of_means_of_E, axis=1)



fig, ax = plt.subplots(2, sharex=True)

ax[0].errorbar(betas, mean_psi, sigma_psi, ls="", marker=".")
ax[1].set_xlabel(r"$\beta$")
ax[0].set_ylabel(r"$\langle m \rangle$")

ax[1].errorbar(betas, mean_H, sigma_H, ls="", marker=".")
ax[1].set_ylabel(r"$\langle E \rangle$")




# ###### CALORE/RISPOSTA
samples_of_chis = np.zeros((n_temperature, n_samples))
samples_of_cvs = np.zeros((n_temperature, n_samples))

for t in track(range(n_temperature), description="resampling chi and cv.."):
    for samp in range(n_samples):
        indexes = np.random.randint(n_samples , size=n_samples)
        fake_sample_m = psi[t, indexes]
        fake_sample_E = H[t, indexes]
        samples_of_chis[t, samp] = 400*( np.mean(fake_sample_m**2)  - np.mean(fake_sample_m)**2)
        samples_of_cvs[t, samp] = 400*( np.mean(fake_sample_E**2)  - np.mean(fake_sample_E)**2)

np.save(f"samples_of_chis.npy",samples_of_chis)
np.save(f"samples_of_cvs.npy",samples_of_cvs)


# samples_of_chis = np.load(f"samples_of_chis.npy")
# samples_of_cvs = np.load(f"samples_of_cvs.npy")

fig, ax = plt.subplots(2, sharex=True)
ax[0].errorbar(betas, np.mean(samples_of_chis, axis=1), np.std(samples_of_chis, axis=1),
                ls="", marker=".")
ax[1].set_xlabel(r"$\beta$")
ax[0].set_ylabel(r"$\chi$")

ax[1].errorbar(betas, np.mean(samples_of_cvs, axis=1), np.std(samples_of_cvs, axis=1),
                ls="", marker=".")
ax[1].set_ylabel(r"$C_v$")

# splot(fig)
colors = sns.color_palette("flare", n_colors=n_temperature)
fig, ax = plt.subplots(2, sharex = True)
for i in range(n_temperature):
    ax[0].plot(autocorr(psi[i])[:100], color=colors[i], label= f"{betas[i]:.2}")
    ax[1].plot(autocorr(H[i])[:100], color=colors[i], label = f"{betas[i]:.2}")
ax[0].set_ylabel(r"$\tau_m$")
ax[1].set_ylabel(r"$\tau_E$")
ax[0].legend(fontsize=6)
# ax[1].legend(fontsize=6)

fig, ax = plt.subplots()

tau_m = np.zeros(n_temperature)
for i in range(n_temperature):
    # print(autocorr(psi[i]))
    tau_m[i] = np.sum(autocorr(psi[i])[:400])
print(tau_m)
ax.plot(betas, tau_m)

fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(psi[19], ls= "", marker= ".", ms=1)
ax[1].plot(H[19],  ls= "", marker= ".", ms=1)
fig.suptitle(rf"$\beta = {betas[19]:.2f}$")

ax[0].set_ylabel("m-chain")
ax[1].set_ylabel("E-chain")

plt.show()