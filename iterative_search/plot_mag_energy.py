import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()
from matplotlib import rcParams
print(rcParams.keys())
rcParams["font.size"] = 10

df = pd.read_csv("euristic_values_120kpoints.csv")
print(df.columns)

fig, (axm, axE) = plt.subplots(2,1, sharex=True, figsize=(7, 4))
colors = {l:c for l, c in zip( np.unique(df.L), sns.color_palette("flare", len(np.unique(df.L))))}

for l in np.unique(df.L):
    subset = df[df.L == l].sort_values("beta").reset_index()
    axm.errorbar(subset.beta, subset.absM_mean, subset.errabsM_mean, ls="-", marker=".", lw=0.3, color=colors[l])
    axE.errorbar(subset.beta, subset.E_mean, subset.errE_mean, ls="-", marker=".", lw=0.3, color=colors[l], label=f"L = {l}")
axm.set_ylabel(r"$\langle |M| \rangle$")
axE.set_ylabel(r"$\langle E \rangle$")
axE.set_xlabel(r"$\beta$")


lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.85, 0.55), ncol=1)

plt.show()