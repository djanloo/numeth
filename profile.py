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


# profile = LineProfiler()
# profile.add_function(ising.ising)
# wrap = profile(ising.ising)
# wrap(100, 5, 0.1, 0.0, 100)
# profile.print_stats()
print(list(ising.this_is_wrong()))

