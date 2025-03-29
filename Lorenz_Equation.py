
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def f(xyz, t, sigma, rho, beta):
    x, y, z= xyz

    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

t = np.linspace(0, 25, 10000)
xyz = [1.0, 1.0, 1.0]
sigma, rho, beta = 8, 28, 8/3.0

xyz = integrate.odeint(f, xyz, t, args=(sigma, rho, beta))

fig, ax = plt.subplots(figsize=(12, 4), subplot_kw={'projection':'3d'})

for i in range(0, len(t), 50):
    ax.plot(xyz[:i, 0], xyz[:i, 1], xyz[:i, 2], 'r', alpha=.5)
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$y$', fontsize=16)
    ax.set_zlabel(r'$z$', fontsize=16)
    ax.set_label("Lorenz Equation")
    plt.pause(0.01)

plt.show()
