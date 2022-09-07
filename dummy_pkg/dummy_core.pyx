"""Dummy module. 

Implements a primes counting algorithm in six different ways."""
# Checks for correct import
from . import dummy_utils, vanilla

# Parallel computing with nogil
from cython.parallel import prange

# Fast c math
from libc.math cimport sqrt

cimport cython

# Checks for correct module content
dummy_utils.urushibara_ruka(1)

### First three algs: check up to n
#1 Just python: uses def
def primes(range_from: int, range_til: int):
  """Most trivial function: checks up to max"""
  prime_count = 0
  range_from = range_from if range_from >= 2 else 2
  for num in range(range_from, range_til + 1):
    for divnum in range(2, num):
      if ((num % divnum) == 0):
        break
    else:
      prime_count += 1
  return prime_count

#2 This will be compiled in c and wrapped in python: uses cpdef
cpdef primes_cy(int range_from, int range_til):
  """ The same as before but with defined types"""
  cdef int prime_count = 0
  cdef int num
  cdef int divnum
  range_from = range_from if range_from >= 2 else 2
  for num in range(range_from, range_til + 1):
    for divnum in range(2, num):
      if ((num % divnum) == 0):
        break
    else:
      prime_count += 1
  return prime_count

#3 This uses the prange parallel generator
cpdef primes_cy_parallel(int range_from, int range_til):
  """ The same as before but parallelised"""
  cdef int prime_count = 0
  cdef int num
  cdef int divnum
  range_from = range_from if range_from >= 2 else 2
  with nogil:
    for num in prange(range_from, range_til + 1, num_threads=4):
      for divnum in range(2, num):
        if ((num % divnum) == 0):
          break
      else:
        prime_count += 1
  return prime_count

# Last three algs: check up to sqrt(n)
# Since the used sqrt is from libc it will be utterly fast

#4 stops at sqrt
def primes_root(range_from: int, range_til: int):
  """Less trivial: checks up to srt(n)"""
  prime_count = 0
  range_from = range_from if range_from >= 2 else 2
  for num in range(range_from, range_til + 1):
    for divnum in range(2, int(num**1/2 + 1)):
      if ((num % divnum) == 0):
        break
    else:
      prime_count += 1
  return prime_count

#5 Stops at libc.math.sqrt
cpdef primes_cy_root(int range_from, int range_til):
  """ The same as before but with defined types and cmath"""
  cdef int prime_count = 0
  cdef int num
  cdef int divnum
  range_from = range_from if range_from >= 2 else 2
  for num in range(range_from, range_til + 1):
    for divnum in range(2, <int> (sqrt(num) + 1) ):
      if ((num % divnum) == 0):
        break
    else:
      prime_count += 1
  return prime_count

#6 Stops at libc.math.sqrt and uses parallel generator
cpdef primes_cy_parallel_root(int range_from, int range_til):
  """ The same as before but parallelised"""
  cdef int prime_count = 0
  cdef int num
  cdef int divnum
  range_from = range_from if range_from >= 2 else 2
  with nogil:
    for num in prange(range_from, range_til + 1, num_threads=4):
      for divnum in range(2, <int> (sqrt(num) + 1)):
        if ((num % divnum) == 0):
          break
      else:
        prime_count += 1
  return prime_count

## Benchmark 2: compiler directives
# Tests if compiler directives given as:
#   - globally (compile using make hardcore)
#   - filewise ( #cython: cdivision=True)
#   - locally (@cython.cdivision(True))
# correctly work

cpdef division(float x, float y, long int number_of_times):
  """Tests the impact of cdivision=True"""
  cdef long int i = 0
  cdef float z = 0 
  for i in range(number_of_times):
    z = x / y
    x += 1.0
    y += 2.0
  return z

@cython.cdivision(True)
@cython.wraparound(False)  # Since no memoryview is used
@cython.boundscheck(False) # those two are useless
cpdef cdivision(float x, float y,long int number_of_times):
  """Tests the impact of cdivision=True"""
  cdef long int i = 0
  cdef float z = 0
  for i in range(number_of_times):
    z = x / y
    x += 1.0
    y += 2.0
  return z

# this is just c: it can't be seen from python
cdef does_nothing():
  pass