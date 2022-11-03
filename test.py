from numeth.core import harmosc

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from numeth.ising import ising
from time import perf_counter

start = perf_counter()
S = ising(N=400, beta=5, J=0.1, h=0.0, N_iter=100)
print(f"Time: {perf_counter()-start}")
plt.matshow(S)
plt.axis("off")
plt.show()