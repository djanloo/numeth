import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

data=pd.read_csv("5_7_10_13_15_200kpoints/euristic_values.csv")
colors = sns.color_palette("flare", n_colors=10)
Ls = np.unique(data.L.values)
markers = ["s", ".", "3", "x", "4"]

fig, (axchi, axcv) = plt.subplots(2, 1, sharex=True)

for l,col in zip(Ls, colors):
    line_set = data.loc[data.L==l].sort_values("beta")
    axchi.plot(line_set.beta, l**2*line_set.chi, color=col, ls="-",alpha=0.3, lw=1)
    axcv.plot(line_set.beta, l**2*line_set.cv, color=col, ls="-",alpha=0.3, lw=1)

    for i in data.iter:
        subset = data.loc[data.L==l].loc[data.iter == i]
        axchi.errorbar(subset.beta, l**2*subset.chi, l**2*subset.errchi, color=col, ls="",marker=".", capsize=0)
        axcv.errorbar(subset.beta, l**2*subset.cv, l**2*subset.errcv, color=col, ls="",marker=".", capsize=0)

axchi.set_ylabel(r"$\chi$")
axcv.set_ylabel(r"$C_v$")
axcv.set_xlabel(r"$\beta$")
plt.show()