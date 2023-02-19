import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


data = pd.read_csv("bootstrapped_longrun.csv")
betas = np.unique(data.beta.values) 
Ls = np.unique(data.L.values)

fig, ax = plt.subplots(2,2, sharex=True)
colors = sns.color_palette("flare", n_colors=7)
style = dict(ls="",capsize=1, marker='.', ms=4, elinewidth=5)

for l, col in zip(Ls, colors):
        subdata = data.loc[data.L == l]
        ax[0,0].errorbar(betas, subdata.m, subdata.err_m, color=col, **style)
        ax[0,1].errorbar(betas, -subdata.e, subdata.err_e, color=col, **style)
        ax[1,0].errorbar(betas, subdata.chi, subdata.err_chi, color=col, **style)
        ax[1,1].errorbar(betas, subdata.cv, subdata.err_cv, color=col, **style)

fig.tight_layout()
plt.show()