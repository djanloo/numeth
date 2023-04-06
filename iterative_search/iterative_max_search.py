import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# CHANGE WORKING DIRECTORY
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from numeth.utils import bootstrap, mp_scheduler, means_and_errors, propose_by_edges
from rich.progress import track
from rich import print
from tabulate import tabulate

EURISTIC_FILE = "euristic_values.csv"
SCHED_FILE = "schedule.csv"

N_SAMPLES = 10_000
CHAIN_THIN = 100
BOOTSTRAP_BINSIZE = 0.05
BOOTSTRAP_RESAMPLES = 10_000
Ls = [20]

PROPOSAL_N_ITER = 5
N_STARTING_BETAS = 8

N_PROCESSES = None

euristic_df = pd.DataFrame()
schedule_df = pd.DataFrame()

# FIRST ITERATION
Ls = np.array(Ls)
betas_0 = np.array([round(u,5) for u in np.linspace(0.35, 0.5, N_STARTING_BETAS)])

for l in Ls:
    for b in betas_0:
        row = dict(iter=0, L=l, beta=b)
        row = pd.DataFrame(row, index=[0])
        schedule_df = pd.concat([schedule_df, row], ignore_index=True)

print("INITIAL SCHEDULE")
print(tabulate(schedule_df, headers="keys", tablefmt="rounded_grid"))

mp_scheduler(schedule_df[schedule_df.iter==0], 
            savefile=f"chain_iter_0.csv", 
            n_samples=N_SAMPLES, 
            n_iters=CHAIN_THIN,
            n_processes=N_PROCESSES)

# DEFINES THE ESTIMATORS
estimators_dict = dict( absM_mean   =["m", lambda x: np.mean(np.abs(x))],
                        E_mean      = ["E", np.mean],
                        chi         =["m", lambda x: np.mean(x**2) - np.mean(np.abs(x))**2],
                        cv          =["E", lambda x: np.mean(x**2) - np.mean(x)**2]
                        )


for it in range(PROPOSAL_N_ITER):

    print(f"\n[bold purple]ITERATION {it} ================================================================== [/bold purple]\n")
    current_iter_chain = pd.read_csv(f"chain_iter_{it}.csv")
    betas, Ls = np.unique(current_iter_chain.beta), np.unique(current_iter_chain.L)
    
    # ESTIMATE CHIs
    this_iter_euristic_df = means_and_errors( schedule_df[schedule_df.iter==it],
                                    current_iter_chain,
                                    estimators_dict, bootstrap_args=dict(    binsize=BOOTSTRAP_BINSIZE, 
                                                                           n_resamples=BOOTSTRAP_RESAMPLES,
                                                                        n_processes=N_PROCESSES)
                                    )
    this_iter_euristic_df["iter"] = it
    euristic_df = pd.concat([euristic_df, this_iter_euristic_df])
    print(f"\n[bold green]EURISTIC ITERATION {it}[/bold green]")
    print(tabulate(euristic_df.loc[euristic_df.iter==it], headers="keys", tablefmt="rounded_grid"))
    
    # PROPOSE NEW BETAS
    for l in Ls:
        subset = euristic_df.loc[euristic_df.L == l].sort_values("beta").reset_index()
        new_betas = propose_by_edges(subset.beta.values, subset.chi.values)
        new_betas = np.around(new_betas, 5)

        for nb in new_betas:
            row = dict(iter=it+1, L=l, beta=nb)
            row = pd.DataFrame(row, index=[0])
            schedule_df = pd.concat([schedule_df, row], ignore_index=True)
    print()
    print(f"\n[bold yellow]SCHEDULE ITERATION {it+1}[/bold yellow]")
    print(tabulate(schedule_df.loc[schedule_df.iter==it+1], headers="keys", tablefmt="rounded_grid"))

    schedule_df.to_csv(SCHED_FILE)
    euristic_df.to_csv(EURISTIC_FILE)


    # RUN THE SCHEDULED SIMULATIONS
    print("\n[bold red]SAMPLING[/bold red]")
    mp_scheduler(schedule_df[schedule_df.iter==it+1], 
                savefile=f"chain_iter_{it+1}.csv", 
                n_iters=CHAIN_THIN, 
                n_samples=N_SAMPLES,
                n_processes=N_PROCESSES)
    print()

sns.lineplot(data=euristic_df, x="beta", y="chi", hue="L")
sns.lineplot(data=euristic_df, x="beta", y="cv", hue="L")

plt.show()
            

        
        
        


