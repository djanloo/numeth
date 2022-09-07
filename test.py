from numeth.core import harmosc
import numpy as np

for pipo in harmosc(np.zeros(6, dtype=np.float32), 0.5, 10000):
    print(pipo)