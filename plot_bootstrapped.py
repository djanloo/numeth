import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


data = pd.read_csv("bootstrapped_all.csv")
betas = np.unique(data.beta.values) 
Ls = np.unique(data.L.values)

colors = sns.color_palette("flare", n_colors=8)
style = dict(ls="",capsize=1, marker='.', ms=4, elinewidth=5)

ax = []
for i in range(4):
        _, a = plt.subplots()
        ax.append(a)
ax = np.array(ax).reshape(-1,2)

for l, col in zip(Ls, colors):
        subdata = data.loc[data.L == l]
        # print(subdata)
        ax[0,0].errorbar(subdata.beta, subdata.m, subdata.err_m, color=col, **style)
        ax[0,1].errorbar(subdata.beta, -subdata.e, subdata.err_e, color=col, **style)
        ax[1,0].errorbar(subdata.beta, subdata.chi, subdata.err_chi, color=col, **style)
        ax[1,1].errorbar(subdata.beta, subdata.cv, subdata.err_cv, color=col, **style)

ax[0,0].set_ylabel(r"$ \langle m \rangle$")
ax[0,1].set_ylabel(r"$E$")
ax[1,0].set_ylabel(r"$\chi$")
ax[1,1].set_ylabel(r"$C_v$")


fig, ax = plt.subplots()
betamax = []

simul_scheduler = pd.DataFrame()
for l, col in zip(Ls, colors):
        subdata = data.loc[data.L == l].reset_index()
        argmax = np.argmax(subdata.chi)
        betamax.append(subdata.loc[argmax].beta)

        new_betas = [   0.5*(betas[argmax-1] + betas[argmax]), 
                        0.5*(betas[argmax+1] + betas[argmax]) ]
        for nb in new_betas:
                row = dict(L=l, beta=nb)
                row = pd.DataFrame(row, index=[0])
                simul_scheduler = pd.concat([simul_scheduler, row], ignore_index=True)

simul_scheduler.to_csv("schedule_iter_1.csv")

        
ax.plot(Ls, betamax , ls="", marker='.')

plt.show()