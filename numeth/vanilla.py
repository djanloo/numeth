"""A module in vanilla python"""
from time import perf_counter
from rich import print
import numpy as np

# Checks if cython stuff can be imported
from . import dummy_utils, dummy_core


class PerfContext:
    """Dummy performance context"""

    def __init__(self, name):
        self.name = name
        self.perfdict = {}
        self.sigmadict= {}
        self.resdict = {}

    def watch(self, func, name="timer", args=None):
        performances = []
        for n in range(5):
            watch_start = perf_counter()
            result = func(*args)
            watch_end = perf_counter()
            performances.append(watch_end - watch_start )
        self.resdict[name] = result
        self.perfdict[name] = np.mean(performances) 
        self.sigmadict[name] = np.std(performances)
        print(".", end="")

    def __enter__(self):
        return self

    def __exit__(self, *args):

        colors = {}
        for name in self.perfdict:
            colors[name] = "blue"

        colors[min(self.perfdict, key=self.perfdict.get)] = "green"
        colors[max(self.perfdict, key=self.perfdict.get)] = "red"

        print()
        for name in self.perfdict:
            print(
                f"[{colors[name]}]{name:<20}[/{colors[name]}] --> ({self.perfdict[name]:8.3f} +- {self.sigmadict[name]:8.3f}) s (result: {self.resdict[name]})"
            )
