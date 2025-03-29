#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import sin, latex
sympy.init_printing()

t, s, Y = sympy.symbols('t, s, Y', real=True)
y = sympy.Function('y')

ode = sympy.Eq(((y(t).diff(t, 2))+(2*y(t).diff(t))+(10*y(t))), (2*sin(3*t)))
L_y = sympy.laplace_transform(y(t), t, s, noconds=True)
ode_rhs_Ly = sympy.laplace_transform(ode.rhs, t, s, noconds=True)
ode_lhs_Ly = sympy.laplace_transform(ode.lhs, t, s, noconds=True)

ode_Ly = sympy.Eq(ode_lhs_Ly.subs(L_y, Y), ode_rhs_Ly)

ics = {y(0):1, y(t).diff(t):0}
Laplace_ode = sympy.Eq((ode_Ly.lhs.subs(ics) - ode_rhs_Ly), 0)
ode_soln = sympy.solve(Laplace_ode, Y)
Solution = sympy.inverse_laplace_transform(ode_soln[0], s, t)

Solution_t0 = Solution.subs({sympy.Heaviside(t):0})
Solution_t1= Solution.subs({sympy.Heaviside(t):1})

Ode_Solution = sympy.lambdify(t, Solution_t1, 'numpy')

T = np.linspace(-5, 5, 100)
xx, yy = np.meshgrid(T, T)
ode_eval = Ode_Solution(T)

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={})

ax.plot(T, ode_eval)
ax.set_title(r'$%s$'%latex(ode))
ax.grid(True)
# %%
