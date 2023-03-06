import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
from scipy.optimize import curve_fit

data=pd.read_csv("euristic_values.csv")
colors = sns.color_palette("flare", n_colors=10)
Ls = np.unique(data.L.values)
markers = ["s", ".", "3", "x", "4"]
M = 5
fig, (axchi, axcv) = plt.subplots(2, 1, sharex=True)
fig, axbetamax = plt.subplots()

def model(beta, a, b,c):
    return -a*beta**2 + b*beta + c
betamax_colors = sns.color_palette("mako", n_colors=10)
for l,col in zip(Ls, colors):
    subset = data.loc[data.L==l]
    order = np.argsort(subset.chi.values)
    axcv.errorbar(subset.beta, l**2*subset.cv, l**2*subset.errcv, color=col, ls="",marker=".", capsize=0)
    for kk, mm in enumerate(range(8,9)):

        best_betas = subset.beta.values[order][-mm:]
        best_chis = l**2*subset.chi.values[order][-mm:]
        best_sigmas = l**2*subset.errchi.values[order][-mm:]

        popt, pars = curve_fit(model, best_betas, best_chis, sigma=best_sigmas)
        print(popt, pars)
        axchi.errorbar(subset.beta, l**2*subset.chi, l**2*subset.errchi, color=col, ls="",marker=".", capsize=0)
        # axchi.errorbar(best_betas, best_chis, best_sigmas, color="k", ls="",marker=".", capsize=0)

        bbetaa = np.linspace(min(best_betas), max(best_betas), 100)
        axchi.plot(bbetaa, model(bbetaa, *popt), color="k")
        
        betamax = 0.5*popt[1]/popt[0]
        errbetamax = betamax*np.sqrt( np.sum( pars.diagonal()/popt**2 ))
        chimax = model(betamax, *popt)
        errchimax = chimax
        axbetamax.errorbar([chimax],[betamax],[errbetamax],ls="",marker=".", color=betamax_colors[kk])

    argm = np.argmax(subset.chi.values)

    axbetamax.scatter([l**2*subset.chi.values[argm]], [subset.beta.values[argm]], marker="3", color="r")
axchi.set_ylabel(r"$\chi$")
axcv.set_ylabel(r"$C_v$")
axcv.set_xlabel(r"$\beta$")
plt.show()