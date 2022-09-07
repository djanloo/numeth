"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from dummy_pkg import dummy_core

# Sets the working directory as the one with the code inside
# Without this line line_profiler won't find anything
chdir(join(dirname(__file__), "dummy_pkg"))

arg = (2,100)

# Comparison between functions that accept the same args
funcs = [dummy_core.primes, 
        dummy_core.primes_cy, 
        dummy_core.primes_cy_parallel, 
        dummy_core.primes_root,
        dummy_core.primes_cy_root,
        dummy_core.primes_cy_parallel_root]

lp = LineProfiler()

for f in funcs:
    lp.add_function(f)
    wrap = lp(f)
    wrap(*arg)

lp.print_stats()