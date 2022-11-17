"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from numeth import ising

# Sets the working directory as the one with the code inside
# Without this line line_profiler won't find anything
chdir(join(dirname(__file__), "numeth"))

profile = LineProfiler()
profile.add_function(ising.stupid_test)
wrap = profile(ising.stupid_test)
wrap()
profile.print_stats()

