import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from numeth.stats import moving_block_bootstrap as mbb
from numeth.utils import means_and_errors, schedule_from_chains

data=pd.read_csv("ising_100.csv")

betas = np.unique(data.beta.values)
pids = np.unique(data.PID.values)

subset = data.loc[data.beta == betas[0]]

for i,p in enumerate(np.unique(subset.PID)):
    plt.plot(np.arange(i*4000, (i+1)*4000),subset.loc[subset.PID == p].m.values, ls="", marker=".", ms=10)

subset = data.sort_values(["beta", "PID"]).drop(columns=["PID",  "iter"]).loc[data.beta == betas[-1]]
ultravar = lambda x: np.mean(x**2) - np.mean(np.abs(x))**2

print("beta", betas[-1])
print("mean(m**2)", np.mean(subset.m.values**2))
print("mean(|m|)**2", np.mean(np.abs(subset.m.values))**2)
print("difference", np.mean(subset.m.values**2) - np.mean(np.abs(subset.m.values))**2)
print("ultravar", ultravar(subset.m.values))
print("mbb", mbb(subset.m.values.astype('float32'), ultravar, n_resamples=100, binsize=0.2))

print(subset.columns)
plt.plot(subset.m.values, lw=1, alpha=0.5)

plt.figure(2)

ultravar = lambda x: np.mean(x**2) - np.mean(np.abs(x))**2

m = np.zeros(len(np.unique(data.beta.values)))
errm = np.zeros(len(betas))
for i,b in enumerate(betas):
    
    values = data.loc[data.beta == b].sort_values("PID").m.values.astype('float32')
    # plt.plot(values + i, ls="", marker=".", ms=2)
    m[i], errm[i] = mbb(
                        values, 
                        ultravar, 
                        n_resamples=1000, binsize=0.01
                        )
    print(i, b, m[i], errm[i])

plt.errorbar(betas, m, errm, ls="", marker=".")
plt.show()

