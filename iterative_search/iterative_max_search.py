import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
 

from numeth.utils import bootstrap, mp_scheduler, means_and_errors, propose_by_centroid
from rich.progress import track

EURISTIC_FILE = "euristic_values.csv"
SCHED_FILE = "schedule.csv"

N_SAMPLES = 10_0
CHAIN_THIN = 10
BOOTSTRAP_BINSIZES = [10,20]
BOOTSTRAP_RESAMPLES = 300
Ls = [10]

PROPOSAL_N_ITER = 5

# euristic_df = pd.read_csv(CHIFILE)
euristic_df = pd.DataFrame()
schedule_df = pd.DataFrame()

# FIRST ITERATION
Ls = np.array(Ls)
betas_0 = np.array([round(u,5) for u in np.linspace(0.35, 0.5, 8)])

for l in Ls:
	for b in betas_0:
		row = dict(iter=0, L=l, beta=b)
		row = pd.DataFrame(row, index=[0])
		schedule_df = pd.concat([schedule_df, row], ignore_index=True)

print("INITIAL SCHEDULE: -----------------------------------------")
print(schedule_df)
print("-----------------------------------------------------------")
mp_scheduler(schedule_df[schedule_df.iter==0], 
			savefile=f"chain_iter_0.csv", 
			n_samples=N_SAMPLES, 
			n_iters=CHAIN_THIN,
			n_processes=4)


for it in range(PROPOSAL_N_ITER):
	print(f"ITERATION: {it} ===================================================")

	current_iter_chain = pd.read_csv(f"chain_iter_{it}.csv")

	betas, Ls = np.unique(current_iter_chain.beta), np.unique(current_iter_chain.L)
	
	# ESTIMATE CHIs
	this_iter_euristic_df = means_and_errors( schedule_df[schedule_df.iter==it],
									current_iter_chain,
									["m"], 
									[lambda x: np.mean(x**2) - np.mean(np.abs(x))**2], 
									["ultravar"], bootstrap_args=dict(bins= BOOTSTRAP_BINSIZES, n_resamples=BOOTSTRAP_RESAMPLES)
									)
	this_iter_euristic_df["iter"] = it
	euristic_df = pd.concat([euristic_df, this_iter_euristic_df])
	print(f"EURISTIC ITERATION {it}: ---------------------------------------------------------")
	print(euristic_df.loc[euristic_df.iter==it])
	print("-----------------------------------------------------------------------------------")
	
	# PROPOSE NEW BETAS
	for l in Ls:
		subset = euristic_df.loc[euristic_df.L == l].sort_values("beta").reset_index()
		new_beta = propose_by_centroid(subset.beta.values, subset.ultravar_m.values)
		new_beta = round(new_beta, 5)

		print(f"L = {l}\tnew_beta = {new_beta}")

		row = dict(iter=it+1, L=l, beta=round(new_beta, 5))
		row = pd.DataFrame(row, index = [0])
		schedule_df = pd.concat([schedule_df, row], ignore_index=True)

	print(f"SCHEDULE ITERATION {it+1}: --------------------------------------------")
	print(schedule_df.loc[schedule_df.iter==it+1])
	print("------------------------------------------------------------------------")
	schedule_df.to_csv(SCHED_FILE)
	euristic_df.to_csv(EURISTIC_FILE)
	# RUN THE SCHEDULED SIMULATIONS
	mp_scheduler(schedule_df[schedule_df.iter==it+1], 
				savefile=f"chain_iter_{it+1}.csv", 
				n_iters=CHAIN_THIN, 
				n_samples=N_SAMPLES,
				n_processes=4)
	
sns.lineplot(data=euristic_df, x="beta", y="ultravar_m", hue="L")
plt.show()
			

		
		
		


