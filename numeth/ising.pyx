"""Module for the ising part"""
from cython.parallel import prange
from libc.math cimport sqrt, log
from libc.stdlib cimport rand, srand
import numpy as np
cimport cython

cdef extern from "gigarand/gigarand.c":
    float idum
    float ran2()

# cdef float ran2_nogil() nogil:
#     return ran2()

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

cdef unsigned int mod(int a, int b):
    """modulo function
    
    Since the % operation can give negative numbers, this brings an integer `a` in the
    interval [0, `b` - 1].
    """
    cdef int r = a % b
    return r if r >= 0 else r + b

# cdef unsigned int mod_nogil(int a, int b) nogil:
#     cdef int r = a % b
#     return r if r >= 0 else r + b

cpdef void set_seed(seed):
    # Sets the time as seed
    idum = <float> seed
    srand(seed)
    return

# def uniformity_test():
#     cdef float [:] array_random = np.zeros(1_000_000, dtype=np.float32)
#     cdef int i
#     for i in range(1_000_000):
#         array_random[i] = ran2()
#     return array_random


def ising(  unsigned int L=10, 
            float beta=0.1, 
            float J=1.0,  
            float h=0.0, 
            unsigned int N_iter=100,
            startfrom=None,
            init="random"):
    
    # Creates the matrix
    cdef int [:, :] S = np.ones((L,L),dtype=np.dtype("i"))

    if startfrom is not None:
        S = startfrom
    elif init == "random":
        S = _ising_random_init(S)

    cdef int proposal, neighborhood, accepted = 0
    cdef unsigned int i, j, iter_index
    cdef float log_r, u

   
    for iter_index in range(N_iter):
        for i in range(L):
            for j in range(L):
                proposal = 2*(rand()%2) - 1
                neighborhood = (
                                S[i, mod(j+1, L)] +
                                S[mod(i+1, L), j] +
                                S[i, mod(j-1, L)] +
                                S[mod(i-1, L), j]
                                )
                log_r = beta*(J*neighborhood + h)*(proposal - S[i,j]) 
                u = ran2()
                if log_r > log(u):
                    accepted += 1
                    S[i,j] = proposal
    return S

cpdef energy(int [:,:] S, float J, float h):
    cdef float H=0
    cdef float H_neigh
    cdef int L = len(S)
    cdef int i,j
    for i in range(L): 
        for j in range(L):
            neighborhood = (S[i, mod(j+1, L)] +S[mod(i+1, L), j] +S[i, mod(j-1, L)] +S[mod(i-1, L), j])
            H_neigh = -((J/4)*(neighborhood)+h)*S[i,j]
            H=H+H_neigh
    return H/L**2

cpdef mag_and_interact(int[:,:] S):
    cdef float interact = 0
    cdef float mag = 0
    cdef float H_neigh
    cdef int L = len(S)
    cdef float Lsq = L**2
    cdef int i,j

    for i in range(L): 
        for j in range(L):
            neighborhood = (S[i, mod(j+1, L)] +S[mod(i+1, L), j] +S[i, mod(j-1, L)] +S[mod(i-1, L), j])
            interact += 0.25*neighborhood*S[i,j]
            mag += S[i,j]/Lsq
    # print(f"returning {mag} {interact}")
    return mag, interact

# cpdef rescale(int [:,:] S):
#     cdef int L = len(S)
#     cdef int [:, :] s = np.ones((L/2,L/2),dtype=np.dtype("i"))
#     cdef int i, j, k, m
#     cdef int n_up = 0
#     for i in range(L/2):
#         for j in range(L/2):
#             # block_sum = S[mod(2*i,L), mod(2*j,L)] + S[mod(2*i + 1,L), mod(2*j,L)] + S[mod(2*i,L), mod(2*j + 1,L)] + S[mod(2*i + 1,L), mod(2*j + 1,L)]
#             for k in [0, 1]:
#                 for m in [0,1]:
#                     if S[mod(2*i + k,L), mod(2*j + m,L)] == 1:
#                         n_up += 1
#             if n_up > 2:
#                 s[i, j] = 1
#             elif n_up < 2:
#                 s[i,j] = -1
#             else:
#                 s[i,j] =  2*(rand()%2) - 1
#             n_up = 0
#     return s
