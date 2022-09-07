from rich import print

# This line rebuilds the cython code
# it is more or less eqivalent to
# >>> import pyximport; pyximport.install()
# with the difference that pyximport sucks

# from dummy_pkg import setup

from dummy_pkg import dummy_core
from dummy_pkg.vanilla import PerfContext

print()

M = 50_000_000
print(f"Fast division benchmark (repeated {M:2.0e} times)")
with PerfContext("division") as p:
    p.watch(dummy_core.division, "normal", args= (200.7, 3.6,  M ))
    p.watch(dummy_core.cdivision, "cdivision", args= (200.7, 3.6, M   ))

print()

N = 100_000
print(f"Primes searching benchmark (up to {N:2.0e}):")
with PerfContext("primes") as p:
    # p.watch(dummy_core.primes, "python", args=(2,N) )
    # p.watch(dummy_core.primes_cy, "cython", args=(2,N) )
    # p.watch(dummy_core.primes_cy_parallel, "cython_parallel", args=(2,N) )
    # p.watch(dummy_core.primes_root, "python_r", args=(2,N) )
    p.watch(dummy_core.primes_cy_root, "cython_r", args=(2,N) )
    p.watch(dummy_core.primes_cy_parallel_root, "cython_parallel_r", args=(2,N) )





