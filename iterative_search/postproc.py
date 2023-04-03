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

NPOINTS = 5000
DISCARD = 3000

a = pd.read_csv("euristic_values_120kpoints.csv").drop(columns=["Unnamed: 0"])

def model(beta, betamax, chimax, A, w):
    return A*(beta-betamax)**2 + chimax

def model_inverse(chi, betamax, chimax, A,w):
    root = np.sqrt((chi - chimax)/A)
    return ( betamax - root, betamax + root)

def log_prior(theta):
    betamax, chimax, A, thr = theta

    if 0<= betamax <= 0.5 and 0 <= chimax <= 1000 and -100_000_000 <= A <= 0 and 0.5 < thr < .95:
        return - np.log(-A)
    else:
        return -np.inf
    
def log_likelihood(theta, beta, chi, yerr):
    betamax, chimax, A, thr = theta
    
    inside_threlsholds_points = chi>thr*max(chi)
    good_betas = beta[inside_threlsholds_points]
    good_errors = yerr[inside_threlsholds_points]
    good_chis = chi[inside_threlsholds_points]
    # if len(good_betas) <= 4:
    #     return -np.inf
    # print("error part", -0.5*np.sum(((good_chis - model(good_betas, *theta))/good_errors)**2))
    # print("threshold part", 0.5*np.sum(inside_threlsholds_points)*np.log(chimax*(thr - 1)/A))
    # print("constant",- 0.5*np.sum(good_errors) )
    return -0.5*np.sum(((good_chis - model(good_betas, *theta))/good_errors)**2) - (len(beta) - np.sum(inside_threlsholds_points))*np.log(2*chimax*thr)  #0.5*len(beta)*np.log(2*(1-thr)) #- 0.5*np.sum(good_errors)

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
fig, axtest2 = plt.subplots()

betamax_global = np.array([])
chimax_global = np.array([])
L_global = np.array([])

results_df = pd.DataFrame()

start_A = {10:-1510, 15:-5371, 20:-14827, 25:-35478, 30:-63913, 40:-187356, 50:-449853, 60:-777138, 80:-1855240, 100:-5_337_469, 120:-15_000_000}

for l in np.unique(a.L):
    subset = a.loc[a.L==l].sort_values("beta").reset_index()
    print(f"Subset with L = {l} has {len(subset)} entries")

    subset.chi = subset.chi*l**2
    subset.errchi = subset.errchi*l**2
    subset.cv = subset.cv*l**2
    subset.errcv = subset.errcv*l**2

    axline.errorbar(subset.beta, subset.chi, subset.errchi, ls="", lw=0.3, marker=".", color=colors[l], capsize=0, elinewidth=1, label=f"L = {int(l)}")
    # axcv.errorbar(subset.beta, subset.cv, subset.errcv, ls="-", lw=0.3, marker=".", color=colors[l], capsize=0, elinewidth=1, label=f"L = {int(l)}")

    # subset = subset.loc[subset.chi > 0.7*np.max(subset.chi)].reset_index()
  
    # STARTING VALUES
    start_chimax = np.max(subset.chi)
    start_betamax = subset.loc[subset.chi.argmax()].beta
    y = subset.sort_values("chi", ascending=False).reset_index(drop=True)

    print(f"L= {l}: starting from betamax={start_betamax}, chimax = {start_chimax}, A = {start_A[l]}")

    # ADDS NOISE
    startpos =  np.zeros((32, 4))
    startpos[:, 0] = np.random.normal(start_betamax, 0.0001, size=32)
    startpos[:, 1] = np.random.normal(start_chimax, 0.0001, size=32)
    startpos[:, 2] = np.random.uniform(start_A[l] - 0.3*abs(start_A[l]), start_A[l] + 0.3*abs(start_A[l]), size=32)
    startpos[:, 3] = np.random.uniform(.6, .9, size=32)

    # SAMPLE
    nwalkers, ndim = startpos.shape
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(subset.beta.values, subset.chi.values, subset.errchi.values))
    sampler.run_mcmc(startpos, NPOINTS, progress=True)

    # GET SAMPLES
    samples = sampler.get_chain()[DISCARD:]

    betamax_samples = samples[:,:, 0].reshape(-1)
    chimax_samples = samples[:,:, 1].reshape(-1)
    A_samples = samples[:,:, 2].reshape(-1)
    w_samples= samples[:,:, 3].reshape(-1)

    mean_betamax, mean_chimax, mean_A = np.mean(betamax_samples),np.mean(chimax_samples), np.mean(A_samples)
    mean_w = np.mean(w_samples)
    print(f"estimates: \nbetamax = {mean_betamax} +- {np.std(betamax_samples)}\nchimax = {mean_chimax}\nA = {mean_A}\nw = {mean_w}")

    ax_test_th.scatter(startpos[:, 3], startpos[:,0], color=colors[l], marker="+", s=50)
    logls = [log_probability(theta,subset.beta.values, subset.chi.values, subset.errchi.values ) for theta in samples.reshape(-1, ndim)]
    logls = np.array(logls)
    logls = (logls - min(logls))/(max(logls) - min(logls))
    ush = ax_test_th.scatter(w_samples, betamax_samples, color=colors[l], marker=".", s=30*logls, alpha=1,)
    ## TEST for threshold
    # for i in range(3):
    #     npoints = np.sum(subset.chi.values > chimax_samples[5*i]*w_samples[5*i])
    #     ax_test_th.scatter([l], [npoints])

    for kkk in range(149):
        betamax, chimax, A, thr = samples.reshape(-1,ndim)[-10*kkk]
        bb = subset.beta.loc[subset.chi > thr*subset.chi.max()]
        minbeta,maxbeta = model_inverse(thr*subset.chi.max(), betamax, chimax, A, thr)
        xx = np.linspace(minbeta,maxbeta, 100)
        axline.plot(xx, model(xx, betamax, chimax, A, thr), lw=1, color=colors[l], alpha=0.05)
    
    axscat.scatter( betamax_samples[::10],chimax_samples[::10], color=colors[l], s=3, alpha=1,label=f"L = {int(l)}")
    axscat.scatter([mean_betamax],[mean_chimax], marker="x", color="k")

    axscaling.scatter((np.ones(len(betamax_samples))*l)[::100],betamax_samples[::100], marker=".",  s=50, color=colors[l], alpha=0.5)

    betamax_global = np.append(betamax_global, betamax_samples)
    chimax_global = np.append(chimax_global, chimax_samples)
    L_global = np.append(L_global, [l]*len(betamax_samples))

    row = dict( L=l,
                betamax_mean=np.mean(betamax_samples[::10]), betamax_err=np.std(betamax_samples[::10]),
                chimax_mean = np.mean(chimax_samples[::10]), chimax_err=np.std(chimax_samples[::10])
            )
    row = pd.DataFrame(row, index=[0])
    results_df = pd.concat([results_df, row], ignore_index=True)

    # plot likelihood al variare di thr
    thr_scan = np.linspace(0.7, 0.99)
    logl = np.zeros(len(thr_scan))
    for i in range(len(thr_scan)):
        theta = [mean_betamax, mean_chimax, mean_A, thr_scan[i]]
        logl[i] = log_probability(theta, subset.beta, subset.chi, subset.errchi )
    axtest2.plot(thr_scan, logl, color=colors[l])


results_df.to_csv("postproc.csv")
axline.legend(ncols=2)
axscat.legend()

axscat.set_xlabel(r"$\beta_{max}$")
axscat.set_ylabel(r"$\chi_{max}$")

axline.set_xlabel(r"$\beta$")
axline.set_ylabel(r"$\chi$")
# axcv.set_ylabel(r"$C_v$")

plt.colorbar(ush, ax= ax_test_th)
# plt.show()

def scaling_func(L, betacrit, nu, a):
    return betacrit + a*L**(-1/nu)

p0 = [0.44, 1.1, 1.6]
popt, pars = curve_fit(scaling_func, results_df.L, results_df.betamax_mean, sigma=results_df.betamax_err, p0=p0)

print(popt)
print(np.sqrt(pars.diagonal()))

ll = np.linspace(min(a.L), max(a.L))

axscaling.plot(ll, scaling_func(ll, *popt), color="k", lw=0.3)
axscaling.axhline(popt[0], lw=1, ls=":", color="k")
axscaling.annotate(rf"$\beta_c = {popt[0]:.4f} \pm {np.sqrt(pars.diagonal())[0]:.4f}$", (50, popt[0]), rotation=0, va="bottom", ha="center")
axscaling.set_ylabel(r"$\beta_{max}$")
axscaling.set_xlabel(r"$L$")
# axscaling.set_xscale("log")
# axscaling.set_yscale("log")

fig, axhyp = plt.subplots()
popt = [0.4402, 0.958, -0.466]


plt.show()