import numpy as np
import pandas as pd
import multiprocessing as mp

from datetime import datetime
from time import perf_counter
from rich.progress import track

from .ising import set_seed, ising, energy


def autocorr(y):
    x = y.copy()
    x -= np.mean(x)
    x /= np.std(x)
    result = np.correlate(x, x, mode='full')/len(x)
    return result[result.size//2:]

def bootstrap(data, function, n_resamples = 500):
    N = len(data)
    bootstrap_distrib = []
    for samp in range(n_resamples):
        indexes = np.random.randint(N , size=N)
        fake_sample = data[indexes]
        bootstrap_distrib.append(function(fake_sample))
    
    return np.mean(bootstrap_distrib), np.std(bootstrap_distrib)

def bin_bootstrap(data, function, n_resamples, bins_sizes):
    N = len(data)
    results = pd.DataFrame(columns = ["bin_size", "estimator_mean", "estimator_error"])
    for i, bs in enumerate(bins_sizes):
        n_blocks = N//bs
        estimators = []
        for samp in range(n_resamples):
            fake_sample = []
            for block_index in range(n_blocks):
                index = np.random.randint(N)
                if index + bs < N:
                    block = data[index: index+bs]
                else:
                    a = index + bs - N
                    block = np.concatenate((data[0:a], data[index:]))
                fake_sample.append(block)
            estimators.append(function(np.array(fake_sample)))
        row = dict(bin_size=bs, estimator_mean= np.mean(estimators), estimator_error= np.std(estimators))
        row = pd.DataFrame(row, index = [0])
        results = pd.concat([results, row], ignore_index=True)
    return results 


def generate(queue, n_samples, ising_params):
    """Single-processor execution of Ising simulation"""
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
        queue.put((PID, i, ising_params["beta"], ising_params["N"], np.mean(S), energy(S, 1.0, 0.0) ))
    queue.put(None)



def mp_scheduler(schedule, savefile="allere_gng.csv", **params):
    """Runs multiprocess Ising with params indicated by schedule"""
    results_df = pd.DataFrame(columns=["PID", "iter", "beta", "L", "m", "E"])

    for index, scheduled_run in schedule.iterrows():

        # Sets the paramteres
        ising_params = dict(N=scheduled_run.L, beta=scheduled_run.beta, N_iter=params["n_iters"], h=0.0)
        n_samples = params["n_samples"]
        q = mp.Queue()

        # Starts the runners
        n_processes = params.get("n_processes", 4)
        runners = [mp.Process(target=generate, args=(q, n_samples, ising_params)) for _ in range(n_processes)]
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
        print("----------------------------------------------------------------------")
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

def means_and_errors(scheduler, chain_df, random_variables, estimators, estimators_names,
                        bootstrap_args = dict(n_resamples = 500, bins = [20,50,100, 200])):
    """Takes a chainfile outputted by mp_scheduler() and estimates stuff.
    returns a dataframe (beta, L, estim1, err_estim1, ...)

    estimators = list of functions to bootstrap
    estimators names ---> guess what
    """
    errnames = ["err_"+en for en in estimators_names]

    analysis = pd.DataFrame()
    for l in np.unique(scheduler.L.values):
        fixed_l = scheduler.loc[scheduler.L == l]
        for b in track(np.unique(fixed_l.beta), description=f"Bootstrapping for for L = {l}"):
            fixed_l_b = fixed_l.loc[fixed_l.beta == b]

            row = dict(L=l, beta=b)

            for rv in random_variables:
                for estimator, est_name, est_errname in zip(estimators, estimators_names, errnames):
                    i_means, i_errs = [], []
                    all_chains = chain_df.loc[(chain_df.L==l)&(chain_df.beta==b)]
                    print(f"Estimated taus for {rv}:")
                    for pid in np.unique(all_chains.PID):
                        independent_run = all_chains.loc[all_chains.PID == pid]
                        tau = np.argmax(np.cumsum(autocorr(independent_run[rv])))
                        print(tau, end = ", ")
                        if max(bootstrap_args["bins"]) < tau:
                            print(f"WARNING (L={l}, beta={b:.2}): max bin size ({max(bootstrap_args['bins'])}) is shorter than autocorrelation time ({tau})")
                        partial = bin_bootstrap(independent_run[rv].values, estimator, bootstrap_args["n_resamples"] ,bootstrap_args["bins"])
                        worst_err_index = np.argmax(partial.estimator_error.values)
                        i_means.append(partial.estimator_mean.values[worst_err_index])
                        i_errs.append(partial.estimator_error.values[worst_err_index])
                    row[est_name+ "_" +rv] = np.mean(i_means)
                    row[est_errname + "_" +rv] = np.sqrt(np.sum(np.array(i_errs)**2))
            row = pd.DataFrame(row, index=[0])
            analysis = pd.concat([analysis, row], ignore_index=True)
    return analysis

def propose_by_centroid(x,p, M=5):
    return np.mean(x[np.argsort(p)][-M:])

# def propose_new_temperature(betas, probabilities, chain_length=50):
#     """Propose a new temperature using a discrete MCMC using chi as a propbability"""

#     index_chain = np.zeros(chain_length)
#     index_chain[0] = np.random.randint(len(betas))

#     for j in range(chain_length - 1):
#         proposal = np.random.randint(len(x))
#         if proposal < 0 or proposal >= len(x):
#             index_chain[j+1] = index_chain[j]
#         elif probabilities[proposal]/probabilities[index_chain[j]]] > np.random.uniform(0, 1):
#             chain.append(proposal)
#         else:
#             chain.append(chain[-1])
#     new_x = np.mean([x[u] for u in chain])
#     x = np.append(x, [new_x])
#     y = np.append(y, [f(new_x)])
#     c = np.append(c, [i+1])