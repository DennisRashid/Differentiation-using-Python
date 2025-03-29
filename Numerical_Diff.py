#%%
import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import sympy
sympy.init_printing()
from Directional_Fields import plot_directional_field

x = sympy.symbols('x')
y = sympy.Function('y')
f = x + y(x)**2
ode = sympy.Eq(y(x).diff(x), f)

f_np = sympy.lambdify((y(x), x), f, 'numpy')
y0 = 0
xp = np.linspace(0, 1.9, 100)
yp = integrate.odeint(f_np, y0, xp)
y_1 = interpolate.interp1d(xp, yp[:, 0], kind='linear')

xm = np.linspace(0, -5, 100)
ym = integrate.odeint(f_np, y0, xm)
y_2 = interpolate.interp1d(xm, ym[:, 0], kind='linear')

xl = np.linspace(-5, 2.2, 100)
yl = integrate.odeint(f_np, y0, xl)
y_3 = interpolate.interp1d(xl, yl[:, 0], kind='linear')

fig, ax = plt.subplots(1, 2, figsize=(12, 4), subplot_kw={})

plot_directional_field(x, y(x), f, ax=ax[0])
ax[0].plot(xm, ym, 'b', lw=2)
ax[0].plot(xp, yp, 'r', lw=2)
ax[0].plot(xl, yl, 'g', lw=2)

x = y= np.linspace(-5, 5, 20)
xx, yy = np.meshgrid(x, y)

Dy = f_np(xx.T, yy.T)
Dx = np.ones_like(Dy)

magnitude = np.sqrt(Dx**2 + Dy**2)
normal_vector_x = Dx / magnitude
normal_vector_y = Dy / magnitude

ax[1].quiver(xx, yy, normal_vector_x, normal_vector_y, angles='xy', scale=30)

ax[1].plot(xm, ym, 'b', lw=2)
ax[1].plot(xp[:-5], yp[:-5], 'r', lw=2)
ax[1].plot(xl[:-1], yl[:-1], 'g', lw=2)
ax[1].set_title(r"$%s$" % sympy.latex(ode))
plt.show()
# %%
