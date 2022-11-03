"""Module for the ising part"""
from cython.parallel import prange
from libc.math cimport sqrt, log
from libc.stdlib cimport rand
import numpy as np
cimport cython

cdef extern from "limits.h":
    int INT_MAX

"""Uniform between 0-1 in nogil mode"""
cdef float randzerone_nogil() nogil:
  return rand()/ float(INT_MAX)

"""Uniform in 0-1 in normal mode"""
cdef float randzerone():
  return rand()/ float(INT_MAX)

cdef int[: ,:] _ising_random_init(int [:, :] S):
    """Random uniform initialization"""
    for i in range(len(S)):
        for j in range(len(S)):
            S[i,j] = 2*(rand()%2) -1
    return S

cdef int mod(int a, int b):
    """modulo function
    
    Since the % operatio can give negative numbers, this brings an integer `a` in the
    interval [0, `b` - 1].
    """
    cdef int r = a % b
    return r if r >= 0 else r + b


def ising(int N=100, float beta=0.1, float J=0.1,  float h=0.1, int N_iter=100, init="random"):
    # Creates the matrix
    cdef int [:, :] S = np.ones((N,N),dtype=np.dtype("i"))
    print(f"N {N}\tNiter {N_iter}\tbeta {beta}\th {h}\tJ {J}")
    if init == "random":
        S = _ising_random_init(S)

    cdef int proposal, neighborhood, i, j, iter_index
    cdef float log_r

    for iter_index in range(N_iter):
        for i in range(N):
            for j in range(N):
                proposal = 2*(rand()%2) - 1
                neighborhood = S[i, mod(j+1, N)] + S[mod(i+1, N), j] + S[i, mod(j-1, N)] + S[mod(i-1, N), j]
                log_r = ( beta*J*neighborhood*(proposal - S[i,j]) # Interaction term
                          + beta*h*(proposal - S[i,j]))           # Field term
                u = randzerone()
                if u == 0.0:
                    print("null probability")
                if log_r > log(u):
                    S[i,j] = proposal
    return S

def this_is_wrong():
    cdef int [:] s = np.ones(4,dtype=np.dtype("i"))
    cdef int i
    i = 0-1
    s[i] = 0
    s[4] = 0
    return s