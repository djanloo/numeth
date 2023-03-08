# CHANGE WORKING DIRECTORY
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from numeth.utils import joinchains
import emcee 

a = pd.read_csv("euristic_values.csv").drop(columns=["Unnamed: 0"])

def model(beta, betamax, chimax, A, w):
    return A*(beta-betamax)**2 + chimax

def log_prior(theta):
    betamax, chimax, A, thr = theta

    if 0<= betamax <= 0.5 and 0 <= chimax <= 20 and -1_000_000 <= A <= 0 and 0 < thr < 0.95:
        return 0.0
    else:
        return -np.inf
    
def log_likelihood(theta, beta, chi, yerr):
    betamax, chimax, A, thr = theta
   
    
    inside_threlsholds_points = chi>thr*max(chi)
    good_betas = beta[inside_threlsholds_points]
    good_errors = yerr[inside_threlsholds_points]
    good_chis = chi[inside_threlsholds_points]
    return -0.5*np.mean(((good_chis - model(good_betas, *theta))/good_errors)**2)

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

colors = {l:c for l,c in zip(np.unique(a.L), sns.color_palette("flare", n_colors=len(np.unique(a.L))))}

fig, axline = plt.subplots()
fig, axscat = plt.subplots()
fig, axscaling = plt.subplots()
fig, ax_test_th = plt.subplots()

betamax_global = np.array([])
chimax_global = np.array([])
L_global = np.array([])

for l in np.unique(a.L):
    subset = a.loc[a.L==l].reset_index()

    subset.chi = subset.chi*l**2
    subset.errchi = subset.errchi*l**2
    subset.cv = subset.cv*l**2
    subset.errcv = subset.errcv*l**2

    axline.errorbar(subset.beta, subset.chi, subset.errchi, ls="", marker=".", color=colors[l], capsize=0, label=f"L = {int(l)}")

    # subset = subset.loc[subset.chi > 0.7*np.max(subset.chi)].reset_index()
  
    # STARTING VALUES
    start_chimax = np.max(subset.chi)
    start_betamax = subset.loc[subset.chi.argmax()].beta
    y = subset.sort_values("chi", ascending=False).reset_index(drop=True)

    start_A = - l**3
    print(f"L= {l}: starting from betamax={start_betamax}, chimax = {start_chimax}, A = {start_A}")

    # ADDS NOISE
    startpos =  np.zeros((32, 4))
    startpos[:, 0] = np.random.normal(start_betamax, 0.001, size=32)
    startpos[:, 1] = np.random.normal(start_chimax, 0.001, size=32)
    startpos[:, 2] = np.random.normal(start_A, 0.03*abs(start_A), size=32)
    startpos[:, 3] = np.random.uniform(.8, .95, size=32)

    # SAMPLE
    nwalkers, ndim = startpos.shape
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(subset.beta.values, subset.chi.values, subset.errchi.values))
    sampler.run_mcmc(startpos, 10000, progress=True)

    # GET SAMPLES
    samples = sampler.get_chain()[5000:]

    betamax_samples = samples[:,:, 0].reshape(-1)
    chimax_samples = samples[:,:, 1].reshape(-1)
    A_samples = samples[:,:, 2].reshape(-1)
    w_samples= samples[:,:, 3].reshape(-1)

    mean_betamax, mean_chimax, mean_A = np.mean(betamax_samples),np.mean(chimax_samples), np.mean(A_samples)
    mean_w = np.mean(w_samples)
    print(f"estimates: \nbetamax = {mean_betamax}\nchimax = {mean_chimax}\nA = {mean_A}\nw = {mean_w}")

    ax_test_th.hist(samples[:,0,3].reshape(-1), histtype="step", bins=40)

    for kkk in range(149):
        betamax, chimax, A, thr = samples.reshape(-1,ndim)[10*kkk]
        bb = subset.beta.loc[subset.chi > thr*subset.chi.max()]
        xx = np.linspace(min(bb), max(bb), 100)
        axline.plot(xx, model(xx, betamax, chimax, A, thr), lw=1, color=colors[l], alpha=0.05)
    # model_samp = [model(xx, *theta) for theta in samples.reshape(-1,ndim)]
    # axline.plot(xx, l**2*model(xx, mean_betamax, mean_chimax, mean_A), color=colors[l])
    # axline.fill_between( xx, *np.quantile(model_samp, [ .02, .98], axis=0), color=colors[l], alpha=0.3)

    axscat.scatter( betamax_samples[::10],chimax_samples[::10], color=colors[l], s=3, label=f"L = {int(l)}")
    axscat.scatter([mean_betamax],[mean_chimax], marker="X", color="k")

    axscaling.scatter([l]*len(betamax_samples), betamax_samples, marker=".",  s=3, color=colors[l])

    betamax_global = np.append(betamax_global, betamax_samples)
    chimax_global = np.append(chimax_global, chimax_samples)
    L_global = np.append(L_global, [l]*len(betamax_samples))

axline.legend()
axscat.legend()

axscat.set_xlabel(r"$\beta_{max}$")
axscat.set_ylabel(r"$\chi_{max}$")

axline.set_xlabel(r"$\beta$")
axline.set_ylabel(r"$\chi$")
plt.show()

def scaling_func(L, betacrit, nu, a):
    return betacrit + a*L**(-1/nu)

p0 = [0.44, 1.1, 1.6]
popt, pars = curve_fit(scaling_func, L_global, betamax_global, p0=p0)

print(popt)
print(np.sqrt(pars.diagonal()))
# print(np.sqrt(pars.diagonal()))
ll = np.linspace(min(a.L), max(a.L))
# axscat.scatter(betamax_global,chimax_global)
# axscat.plot(bb, scaling_func(bb, *p0), color="k", lw=0.3)
axscaling.plot(ll, scaling_func(ll, *popt), color="k", lw=0.3)

plt.show()