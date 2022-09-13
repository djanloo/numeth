"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from numeth import core
import numpy as np

# Sets the working directory as the one with the code inside
# Without this line line_profiler won't find anything
chdir(join(dirname(__file__), "numeth"))

arg = (np.zeros(200, dtype=np.float32), 1.0, 1000)


# Comparison between functions that accept the same args
funcs = [core.harmosc, core.dummy_last_2]

lp = LineProfiler()

for f in funcs:
    lp.add_function(f)
    wrap = lp(f)
    wrap(*arg)



lp.print_stats()