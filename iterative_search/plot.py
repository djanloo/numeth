import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

data=pd.read_csv("euristic_values.csv")
colors = sns.color_palette("flare", n_colors=10)
Ls = np.unique(data.L.values)
for l,col in zip(Ls, colors):
    subset = data.loc[data.L==l]
    plt.errorbar(subset.beta, subset.ultravar_m, subset.err_ultravar_m, color=col, ls="",marker=".")

plt.show()