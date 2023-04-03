# Teste renormalization
from time import time
import numpy as np
import matplotlib.pyplot as plt 
from numeth.ising import ising, set_seed, mag_and_interact, rescale
from numeth.utils import renormalize_mp

ising_params= dict(L=512, beta=.1, N_iter=500)
df = renormalize_mp(10, ising_params)
plt.scatter(df.interact/df.L**2, df.mag, ls="", marker=".", c=df.L, cmap="viridis")

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()