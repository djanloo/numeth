import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

data=pd.read_csv("euristic_values.csv")
colors = sns.color_palette("flare", n_colors=10)
Ls = np.unique(data.L.values)
markers = ["s", ".", "3", "x", "4"]

for l,col in zip(Ls, colors):
    for i in data.iter:
        subset = data.loc[data.L==l].loc[data.iter == i]
        plt.errorbar(subset.beta, l**2*subset.ultravar_m, l**2*subset.err_ultravar_m, color=col, ls="",marker=markers[i], capsize=0)
plt.ylabel(r"$\chi$")
plt.xlabel(r"$\beta$")
plt.show()