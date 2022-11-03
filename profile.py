"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from numeth import core
from numeth import ising
import numpy as np

# Sets the working directory as the one with the code inside
# Without this line line_profiler won't find anything
chdir(join(dirname(__file__), "numeth"))

# lp.add_function(f)
#     wrap = lp(f)
#     wrap(*arg)

profile = LineProfiler()
profile.add_function(ising.ising)
profile.add_function(ising.mod)
wrap = profile(ising.ising)
wrap(50, 5, 0.1, 0.0, 50)
profile.print_stats()

