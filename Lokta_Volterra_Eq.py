#%%
import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import sympy
sympy.init_printing()

a, b, c, d = 0.4, 0.002, 0.001, 0.7

def f(xy, t):
    x, y = xy
    return [a*x - b*x*y, c*x*y - d*y]

xy0 = [600, 400]
t = np.linspace(0, 50, 250)
xy_t = integrate.odeint(f, xy0, t)
xy_prey = interpolate.interp1d(t, xy_t[:,0], kind='linear')
xy_predator = interpolate.interp1d(t, xy_t[:, 1], kind='linear')

fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={})

axes[0].plot(t, xy_t[:, 0], 'b', label='Prey')
axes[0].plot(t, xy_t[:, 1], 'r', label='Predator')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Number of animals')
axes[0].legend()

axes[1].plot(xy_t[:, 0], xy_t[:, 1], 'k')
axes[1].set_xlabel('Number of preys')
axes[1].set_ylabel('Number of pradator')
# %%
