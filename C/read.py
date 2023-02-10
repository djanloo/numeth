import numpy as np
from matplotlib import pyplot as plt

S = np.loadtxt("ising.data").reshape(100,100,10000)

plt.matshow(S[:,:,-1])
plt.show()