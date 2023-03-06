import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import numpy as np
import pandas as pd

# CHANGE WORKING DIRECTORY
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from numeth.stats import get_bootstrap_samples
from scipy.stats import kstat
chain = pd.read_csv("chain_iter_0.csv")

betas = np.unique(chain.beta)
Ls = np.unique(chain.L)

print("beta",betas)
print("L",Ls)

chi_func = lambda x: np.mean(x**2) - np.mean(np.abs(x))**2
for b in betas:
    subsample = chain.loc[chain.beta == b].loc[chain.L==Ls[-1]]
    chi = get_bootstrap_samples(subsample.m.values.astype("float32"), chi_func, n_resamples=10_000)

    # plt.hist(chi, histtype="step", bins=int(4*np.log(len(chi))))
    mean = np.mean(chi)
    plt.scatter([b,], [kstat(chi, 4)] )
plt.show()