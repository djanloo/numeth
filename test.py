from numeth.core import harmosc
import numpy as np
import matplotlib.pyplot as plt

v = np.array(harmosc(np.zeros(2000, dtype=np.float32), 1.0, 10000))
for pipo in v:
    print(pipo)

cazzo cazzetto

plt.plot(v)
plt.show()

