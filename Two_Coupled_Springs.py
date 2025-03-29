#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def f(t, y, args):
    m1, k1, g1, m2, k2, g2 = args

    return [y[1], (k2/m1)*(y[2]-y[0])-(k1/m1)*y[0]-(g1/m1)*y[1], y[3], (-g2/m2)*y[3]-(k2/m2)*(y[2]-y[0])]
    
m1, k1, g1 = 1.0, 10.0, 0.5
m2, k2, g2 = 2.0, 40.0, 0.25
args = (m1, k1, g1, m2, k2, g2)
y0 = [1.0, 0, 0.5, 0]
t = np.linspace(0, 20, 1000)

r = integrate.ode(f)
r.set_integrator('lsoda')
r.set_initial_value(y0, t[0])
r.set_f_params(args)

dt = t[1] - t[0]
y = np.zeros((len(t), len(y0)))
idx = 0

while r.successful() and r.t<t[-1]:
    y[idx, :] = r.y
    r.integrate(r.t + dt)
    idx += 1


fig=plt.figure(figsize=(12, 6))
ax1= plt.subplot2grid((2, 5), (0, 0), colspan=3)
ax2= plt.subplot2grid((2, 5), (1, 0), colspan=3)
ax3= plt.subplot2grid((2, 5), (0, 3), colspan=2, rowspan=2)

for i in range(0, len(t), 100):
    ax1.plot(t[:i], y[:i, 0], 'r')
    ax1.set_ylabel("$x_1$", fontsize=18)
    ax1.set_yticks([-1, -.5, 0, .5, 1])
    ax2.plot(t[:i], y[:i, 2], 'b')
    ax2.set_ylabel('$x_2$', fontsize=18)
    ax2.set_xlabel('t', fontsize=18)
    ax2.set_yticks([-1, -.5, 0, .5, 1])
    ax3.plot(y[:i, 0], y[:i, 2], 'k')
    ax3.set_ylabel('$x_2$', fontsize=18)
    ax3.set_xlabel('$x_1$', fontsize=18)
    ax3.set_xticks([-1, -.5, 0, .5, 1])
    ax3.set_yticks([-1, -.5, 0, .5, 1])
    plt.pause(0.5)

plt.show()
# %%
