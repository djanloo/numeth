from numeth.core import harmosc
import numpy as np
import matplotlib.pyplot as plt

v = np.array(harmosc(np.zeros(100, dtype=np.float32), 0.01, 10000))
for pipo in v:
    print(pipo)

plt.plot(v)
plt.show()

