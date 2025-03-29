#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def f(xyz, t, sigma, rho, beta):
    x,y,z = xyz
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

sigma, rho, beta = 8, 28, 8/3.0
t = np.linspace(0, 25, 10000)
xyz0 = [1.0, 1.0, 1.0]

xyz1 = integrate.odeint(f, xyz0, t, args=(sigma, rho, beta))
xyz2 = integrate.odeint(f, xyz0, t, args=(sigma, rho, 0.6*beta))
xyz3 = integrate.odeint(f, xyz0, t, args=(2.0*sigma, rho, 0.6*beta))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection':'3d'})

for i in range(0, len(t), 50):
    for ax, xyz, c in [(ax1, xyz1, 'r'), (ax2, xyz2, ' b'), (ax3, xyz3, 'g')]:
        ax2.plot(xyz2[:i, 0], xyz2[:i, 1], xyz2[:i, 2], 'b', alpha=.5)
        ax.plot(xyz[:i, 0], xyz[:i, 1], xyz[:i, 2], c, alpha=.5)
        ax.set_xlabel(r'$x$', fontsize=16)
        ax.set_ylabel(r'$y$', fontsize=16)
        ax.set_zlabel(r'$z$', fontsize=16)
        ax.set_xticks([-15, 0, 15])
        ax.set_yticks([-20, 0, 20])
        ax.set_zticks([0, 20, 40])
        plt.pause(0.1)

plt.show()
# %%
