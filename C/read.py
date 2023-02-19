import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
def autocorr(y):
    x = y.copy()
    x -= np.mean(x)
    x /= np.std(x)
    result = np.correlate(x, x, mode='full')/len(x)
    return result[result.size//2:]

N = 20
T = 10_000
S = np.loadtxt("data/conf_h_0.00_beta_0.10_L_20_T_10000.data").reshape(T, N,N)

fig, ax = plt.subplots(2,1)
m = np.mean(np.mean(S, axis=1), axis=1)
ax[0].plot(m)
ax[1].plot(autocorr(m))

fig, ax = plt.subplots()
ax.matshow(S[0, :,:])
ax.set_title("snapshot")

fig, ax = plt.subplots()
matsh = ax.matshow(S[0, :,:])

def update(i):
    matsh.set_data(S[i, :,:])
    print(f"set {i}")

anim = FuncAnimation(fig, update, interval=1000/60,frames=800)

plt.show()