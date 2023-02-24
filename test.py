import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from numeth.stats import moving_block_bootstrap as mbb
from numeth.utils import means_and_errors, schedule_from_chains
from tabulate import tabulate

BOOTSTRAP_BINSIZE = 0.05
BOOTSTRAP_RESAMPLES = 1000

data=pd.read_csv("ising_100.csv")
print(tabulate(data.loc[:3], headers="keys", tablefmt="rounded_grid"))
print(f"max iter = {max(data.iter)}")

iter_thresholds =  [100 ,300, 1000, 2000, 3000, 3999]
colors = sns.color_palette("flare", n_colors=len(iter_thresholds))
fig, (axchi, axcv) = plt.subplots(2,1)
for it_t, col in zip(iter_thresholds, colors):
    cutted = data.loc[data.iter <= it_t]
    euristic = means_and_errors(     schedule_from_chains(cutted),
                                        cutted,
                                        ["m", "E"], 
                                        [lambda x: np.mean(x**2) - np.mean(np.abs(x))**2], 
                                        ["ultravar"], bootstrap_args=dict(binsize=BOOTSTRAP_BINSIZE, n_resamples=BOOTSTRAP_RESAMPLES)
                                        )
    axchi.errorbar(euristic.beta, euristic.ultravar_m, euristic.err_ultravar_m , ls="", marker=".", color=col, capsize=0, label=f"{it_t}")
    axcv.errorbar(euristic.beta, euristic.ultravar_E, euristic.err_ultravar_E , ls="", marker=".", color=col, capsize=0, label=f"{it_t}")

plt.legend()
plt.show()
print(tabulate(euristic, headers="keys", tablefmt="rounded_grid"))

