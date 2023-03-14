import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.optimize import curve_fit

def scaling_func(L, betacrit, nu, a):
    return betacrit + a*L**(-1/nu)

fit_values = pd.read_csv("postproc.csv")
fig, ax = plt.subplots()

ax.errorbar(fit_values.L, fit_values.betamax_mean, fit_values.betamax_err, ls="", marker=".", capsize=0)

p0 = [0.44, 1.1, 1.6]
popt, pars = curve_fit(scaling_func, fit_values.L, fit_values.betamax_mean, sigma=fit_values.betamax_err, p0=p0)

print(popt)
print(np.sqrt(pars.diagonal()))
# print(np.sqrt(pars.diagonal()))
ll = np.linspace(min(fit_values.L), max(fit_values.L))
# axscat.scatter(betamax_global,chimax_global)
# axscat.plot(bb, scaling_func(bb, *p0), color="k", lw=0.3)
ax.plot(ll, scaling_func(ll, *popt), color="k", lw=0.3)

plt.show()