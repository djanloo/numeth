import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from numeth.utils import bootstrap

from rich.progress import track

experim = pd.read_csv("ising_100.csv")
betas = np.unique(experim.beta.values)
Ls = np.unique(experim.L.values)

bootstrap_df = pd.DataFrame(columns = ["L", "beta", "m", "err_m", "chi", "err_chi",  "e", "err_e", "cv", "err_cv"])
for L in Ls:
    for beta in track(betas, description=f"L = {L}"):

        row = dict(L=L, beta=beta)

        data = np.abs(experim.loc[(experim.L==L)&(experim.beta==beta)].m.values)
        row["m"], row["err_m"] = bootstrap(data, lambda x: np.mean(x), n_resamples=100)
        row["chi"], row["err_chi"] = bootstrap(data, lambda x: L**2*(np.mean(x**2) - np.mean(x)**2), n_resamples=100)
        
        data = np.abs(experim.loc[(experim.L==L)&(experim.beta==beta)].E.values)
        row["e"], row["err_e"] = bootstrap(data, lambda x: np.mean(x), n_resamples=100)
        row["cv"], row["err_cv"] = bootstrap(data, lambda x: L**2*(np.mean(x**2) - np.mean(x)**2), n_resamples=100)
        
        block_df = pd.DataFrame(row, index=[0])
        
        bootstrap_df = pd.concat([bootstrap_df, block_df], ignore_index=True)

    bootstrap_df.to_csv("bootstrapped_100.csv")