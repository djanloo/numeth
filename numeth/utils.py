import os 
import multiprocessing as mp

import numpy as np
import pandas as pd

from datetime import datetime
from time import perf_counter
from rich.progress import track
from rich import print
from tabulate import tabulate

from .ising import set_seed, ising, energy, mag_and_interact, rescale
from.stats import moving_block_bootstrap as mbb

def autocorr(y):
    x = y.copy()
    x -= np.mean(x)
    x /= np.std(x)
    result = np.correlate(x, x, mode='full')/len(x)
    return result[result.size//2:]

def estimate_tau(x):
    return np.argmax(np.cumsum(autocorr(x)))

def bootstrap(data, function, n_resamples = 500):
    N = len(data)
    bootstrap_distrib = []
    for samp in range(n_resamples):
        indexes = np.random.randint(N , size=N)
        fake_sample = data[indexes]
        bootstrap_distrib.append(function(fake_sample))
    
    return np.mean(bootstrap_distrib), np.std(bootstrap_distrib)

def generate(queue, n_samples, ising_params):
    """Single-processor execution of Ising simulation"""
    PID = mp.current_process().pid
    set_seed(PID)

    # Sets the parameters for thermal burn in
    burn_in_params = ising_params.copy()
    burn_in_params["N_iter"] = 5000 
    
    S = ising(**burn_in_params)     # THERMALIZATION
   
    for i in range(n_samples):
        ising(startfrom=S, **ising_params)
        queue.put((PID, i, ising_params["beta"], ising_params["L"], np.mean(S), energy(S, 1.0, 0.0) ))
    queue.put(None)



def mp_scheduler(schedule, savefile="allere_gng.csv", **params):
    """Runs multiprocess Ising with params indicated by schedule"""
    results_df = pd.DataFrame(columns=["PID", "iter", "beta", "L", "m", "E"])

    for index, scheduled_run in schedule.iterrows():
        start = perf_counter()
        # Sets the paramteres
        ising_params = dict(L=scheduled_run.L, 
                            beta=scheduled_run.beta, 
                            N_iter=params["n_iters"], 
                            h=0.0)

        n_samples = params["n_samples"]
        q = mp.Queue()

        # Starts the runners
        n_processes = params.get("n_processes", mp.cpu_count())
        runners = [mp.Process(target=generate, args=(q, n_samples, ising_params)) for _ in range(n_processes)]
        for p in runners:
            p.start()

        time_info = datetime.now()
        infos = dict(   time=f"{time_info.hour:2}:{time_info.minute:2}:{time_info.second:2}", 
                        L=ising_params["L"],
                        beta=ising_params["beta"],
                        n_samples=params["n_samples"],
                        n_iter = params["n_iters"],
                        n_processes=n_processes)

        print(tabulate([infos], tablefmt="rounded_grid", headers="keys"), end="")
        

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

            if finished == n_processes:
                break

        for p in runners:
            p.join()

        partial = pd.DataFrame(results, columns=["PID", "iter", "beta","L", "m", "E"])
        results_df = pd.concat([results_df, partial], ignore_index=True)
        print(f" >>>>> {perf_counter() - start:.1f} seconds")
    results_df.to_csv(savefile)

def schedule_from_chains(chains_df):
    scheduler = pd.DataFrame()
    for l in np.unique(chains_df.L):
        fixed_l = chains_df.loc[chains_df.L==l]
        for b in np.unique(fixed_l.beta):
            row = dict(L=l, beta=b)
            row = pd.DataFrame(row, index= [0])
            scheduler = pd.concat([scheduler, row], ignore_index=True)
    return scheduler

def means_and_errors(scheduler, chain_df, estimators_dict,
                        bootstrap_args = None):
    """Takes a chainfile outputted by mp_scheduler() and estimates stuff.
    returns a dataframe (beta, L, estim1, err_estim1, ...)

    estimators_dict = {"estimator_name":[variable, function]}

    """
    

    analysis = pd.DataFrame()
    for l in np.unique(scheduler.L.values):
        fixed_l = scheduler.loc[scheduler.L == l]
        for b in track(np.unique(fixed_l.beta), description=f"Bootstrapping for for L = {l}"):
            # Concatenates chains by pid value
            concatenated_chains = chain_df.loc[(chain_df.L==l)&(chain_df.beta==b)].sort_values("PID")  
            row = dict(L=l, beta=b)

            for est_name in estimators_dict.keys():
                rv , estimator = estimators_dict[est_name]
                row[est_name], row["err" + est_name] = mbb(concatenated_chains[rv].values.astype('float32'), 
                                                                        estimator, 
                                                                        **bootstrap_args
                                                                        )
            row = pd.DataFrame(row, index=[0])
            analysis = pd.concat([analysis, row], ignore_index=True)
    return analysis

def propose_by_centroid(x,p, M=5):
    return np.mean(x[np.argsort(p)][-M:])


def propose_by_edges(x, p, n_edges=2):
    x, p= np.array(x), np.array(p)

    sort = np.argsort(x)
    x = x[sort]
    p = p[sort]

    argmax = np.argmax(p)
    new_xs = []
    for i in range(n_edges):
        if argmax >= 1 + i:
            new_xs.append( 0.5*(x[argmax - (i)] + x[argmax-(i+1)]) )
        if argmax < len(x) - (1 + i):
            new_xs.append( 0.5*(x[argmax + (i)] + x[argmax+(1+i)]) )
    return np.array(new_xs)

def joinchains(dir):
    joined_chains = pd.DataFrame()
    this_iter = 0
    while True:
        try:
            iter_chain = pd.read_csv(f"chain_iter_{this_iter}.csv").drop(columns="Unnamed: 0")
            # print(iter_chain.columns)
            iter_chain = iter_chain.rename(columns={"iter":"point_number"})
            # print(iter_chain.columns)
            iter_chain["iter"] = this_iter
            joined_chains = pd.concat([joined_chains, iter_chain], ignore_index=True)
            this_iter += 1
        except FileNotFoundError:
            break
    return joined_chains

def _renormalize_worker(queue, n_points, ising_kwargs):
    
    n_renorm = int(np.log2(ising_kwargs["L"]) - 2)

    PID = mp.current_process().pid
    set_seed(PID)
    burn_in = ising_kwargs.copy()
    burn_in["N_iter"] = 3000
    S = ising(**burn_in)

    for i in range(n_points):
        ising(**ising_kwargs, startfrom=S)
        s = S.copy()
        for j in range(n_renorm):
            queue.put((len(s),) + mag_and_interact(s))
            s = rescale(s)
    queue.put(None)

def renormalize_mp(n_points, ising_kwargs):

    q = mp.Queue()

    # Starts the runners
    n_processes = mp.cpu_count()
    runners = [mp.Process(target=_renormalize_worker, args=(q, n_points, ising_kwargs)) for _ in range(n_processes)]
    for p in runners:
        p.start()
    

    results = []
    finished = 0

    while True:
        item = q.get()

        # Listens to end of computations
        if item is None:
            finished += 1 
        else:
            results.append(item)
            print(item)

        if finished == n_processes:
            break

    for p in runners:
        p.join()
    
    results_df = pd.DataFrame(results, columns=["L", "mag", "interact"])
    return results_df
