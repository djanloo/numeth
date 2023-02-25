from cython.parallel import prange
from libc.math cimport sqrt, log
from libc.stdlib cimport rand, srand
import numpy as np
cimport cython

import multiprocessing as mp

cdef extern from "gigarand/gigarand.c":
    float idum
    float ran2()

def _moving_block_bootstrap_(float [:] x,
                            estimator,
                            queue, 
                            n_resamples,
                            block_size_percentage,
                            ):
                            
    # Set the random seed using pid
    srand(mp.current_process().pid)

    cdef int N = len(x)
    cdef int block_size = int(N*block_size_percentage)


    cdef int i, block_startpoint, sample_number
    cdef int M = N//int(block_size)                          # Number of blocks
    
    cdef float [:] fake_samp = np.zeros(N, dtype='float32')              # Fake sample
    cdef float [:] estimator_samples = np.zeros(n_resamples, dtype='float32')

    for sample_number in range(n_resamples):
        for i in range(M):
            block_startpoint = rand()%(N - block_size - 1)
            fake_samp[block_size*(i):block_size*(i+1)] = x[block_startpoint:block_startpoint + block_size]
        queue.put(estimator(np.array(fake_samp)))
    queue.put(None)
        

def moving_block_bootstrap(float [:] x, estimator, n_processes=4, n_resamples=1000, binsize=0.2):

    q = mp.Queue()
    # Starts the runners
    runners = [mp.Process(  target=_moving_block_bootstrap_, 
                            args=(x, estimator, q, n_resamples//n_processes, binsize) ) for _ in range(n_processes)]
    for p in runners:
        p.start()

    estimator_samples = []

    ended_processes = 0
    while ended_processes < n_processes:

        item = q.get()
        if item is None:
            ended_processes += 1
        else:
            estimator_samples.append(item)
    for p in runners:
            p.join()
    return np.mean(estimator_samples), np.std(estimator_samples)



    
    
