# cython: boundscheck=False
"""Module for core calculations"""
from cython.parallel import prange
from libc.math cimport sqrt, log
from libc.stdlib cimport rand

# cimport cython
cdef extern from "limits.h":
    int INT_MAX

cdef float randelta() nogil:
  return 2.0*( rand()/ float(INT_MAX) - 0.5 )

cdef float randzerone() nogil:
  return rand()/ float(INT_MAX)

cpdef harmosc(float [:] x, float eps, int refresh, float eta=0.1):
  """Commento a caso"""
  cdef int N = len(x)
  cdef int k = 0
  cdef float proposal = 0.0, log_r = 0.0
  for i in range(refresh):
    with nogil:
      # Note that range(N) sweeps the interval [0, ..., N-1]
      # ciclicity is imposed using %N
      # x*x is used instead of x**2 for speed
      for k in prange(N):
        proposal = x[k] + eps*randelta()
        log_r = -(proposal*proposal - x[k]*x[k])*(1.0/eta + eta/2.0) + 1.0/eta * (proposal - x[k])*(x[(k+1)%N] + x[(k-1)%N])
        if log_r > log(randzerone()):
          x[k] = proposal
  return x

cpdef dummy_last(float [:] x, float eps, int refresh):
  return 0.0

cpdef dummy_last_2(float [:] x, float eps, int refresh):
  return 2.3

cpdef dummy_no_memviews(float x):
  cdef int i 
  for i in range(10):
    x+=randzerone()
  return x