"""Module for core calculations"""
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt
from libc.stdlib cimport rand

cdef extern from "limits.h":
    int INT_MAX

cdef float randelta() nogil:
  return 2.0*( rand()/ float(INT_MAX) - 0.5 )

cdef float randzerone() nogil:
  return rand()/ float(INT_MAX)



cdef float log_prob(float x_previous, float x_current) nogil:
  cdef float deltax = x_current - x_previous
  return -(x_current*x_current + deltax*deltax) 


cpdef harmosc(float [:] x, float eps, int refresh):
  cdef int N = len(x)
  cdef int k = 0
  cdef float proposal = 0.0
  for i in range(refresh):
    with nogil:
      for k in prange(1, N):
        proposal = x[k] + eps*randelta()
        if log_prob(x[k-1], x[k]) < randzerone():
          x[k] = proposal
  return x
  
