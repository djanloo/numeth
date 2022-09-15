from numeth.core import harmosc

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

T = .1
etas = [.01, .005, .001, 1e-4]
mean_squared_x = []
mean_squared_delta = []
for eta in etas:
    ti = perf_counter()
    N = int(1.0/(T * eta))
    x = np.random.rand(N).astype(np.float32)
    x = np.array(harmosc(x,  1.0, 500 , eta=eta))
    mean_squared_x.append(np.mean(x**2))
    mean_squared_delta.append(np.mean(np.diff(x)**2))
    print(f"eta = {eta} --> N = {N} --> time = {perf_counter()-ti}")
    plt.plot(x, ls="", marker=".")
print(f"U = {0.5*np.array(mean_squared_x)  - 0.5/(np.array(etas))**2 *np.array(mean_squared_delta) }")
plt.figure(2)
plt.plot(etas,mean_squared_x, ls="", marker=".")
plt.show()

