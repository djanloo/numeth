import numpy as np
from rich.progress import track

def autocorr(y):
    x = y.copy()
    x -= np.mean(x)
    x /= np.std(x)
    result = np.correlate(x, x, mode='full')/len(x)
    return result[result.size//2:]

def bootstrap(sample, function, n_resamples = 500):
    N = len(sample)
    bootstrap_distrib = []
    for samp in range(n_resamples):
        indexes = np.random.randint(N , size=N)
        fake_sample = sample[indexes]
        bootstrap_distrib.append(function(fake_sample))
    
    return np.mean(bootstrap_distrib), np.std(bootstrap_distrib)