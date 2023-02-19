import numpy as np
import multiprocessing as mp
from numeth.ising import ising, energy, set_seed
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from datetime import datetime
from time import perf_counter
from rich.progress import track
sns.set()

def generate(queue, n_samples, ising_params):

    PID = mp.current_process().pid

    # Sets the parameters for thermal burn in
    burn_in_params = ising_params.copy()
    burn_in_params["N_iter"] = 1000 

    # Set the parameters that prevent from parallel-chain flip
    burn_in_biasing_params = ising_params.copy()
    burn_in_biasing_params["h"] = 1.0
    burn_in_params["N_iter"] = 1000 
    
    set_seed(PID)

    time_info = datetime.now()
    print(f"{time_info.hour}:{time_info.minute}:{time_info.second} >>> PID {PID} " + "\t".join([f"{key}: {value}" for key, value in ising_params.items()]))
    
    S = ising(**burn_in_biasing_params)     # bias burn-in
    ising(**burn_in_params, startfrom=S)    # thermal burn-in
   
    for i in range(n_samples):
        ising(startfrom=S, **ising_params)
        # print(f"PID: {PID}, i={i}, m={np.mean(S)}")
        queue.put((PID, i, ising_params["beta"], ising_params["N"], np.mean(S), energy(S, 1.0, 0.0) ))
    queue.put(None)

##################################### MAIN #####################################


results_df = pd.DataFrame(columns=["PID", "iter", "beta", "L", "m", "E"])

betas = np.linspace(0.35, 0.5, 30)
temps_done = 0

for beta in betas:
    start = perf_counter()
    for N in [100]:

        # Sets the paramteres
        ising_params = dict(N=N, beta=beta, N_iter=100, h=0.0)
        n_samples = 4000
        q = mp.Queue()

        # Starts the runners
        runners = [mp.Process(target=generate, args=(q, n_samples, ising_params)) for _ in range(4)]
        for p in runners:
            p.start()

        # Listens to the runners
        results = []
        finished = 0
        while True:
            item = q.get()

            # Listens to end of computations
            if item is None:
                finished += 1 
            else:
                results.append(item)

            if finished == 4:
                break

        for p in runners:
            p.join()

        partial = pd.DataFrame(results, columns=["PID", "iter", "beta","L", "m", "E"])
        results_df = pd.concat([results_df, partial], ignore_index=True)
    end = perf_counter()
    temps_done += 1
    print(f"---------------- {end-start:.2f} sec -- remaining {(end-start)*(len(betas) - temps_done)/60:.2f} minutes ----------------")
results_df.to_csv("ising_100.csv")
