#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy
sympy.init_printing()

t, omega0, gamma = sympy.symbols('t, omega_0, gamma')
X = sympy.Function('x')

ode = sympy.Eq(((X(t).diff(t, 2)) + (2*gamma*omega0*(X(t).diff(t))) + (omega0**2*X(t))), 0)
ode_soln = sympy.dsolve(ode)

ics = {X(t).subs(t, 0):1, X(t).diff(t).subs(t, 0):0}
def apply_ics(sol, ics, x, known_params):
    free_params = sol.free_symbols - set(known_params)
    eq = [((sol.lhs.diff(x, n)- sol.rhs.diff(x, n)).subs(x, 0).subs(ics)) for n in range(len(ics))]
    soln_eq = sympy.solve(eq, free_params)
    return sol.subs(soln_eq)

solution = apply_ics(ode_soln, ics, t, [omega0, gamma])
x_critical = sympy.limit(solution.rhs, gamma, 1)
w0 = 2*sympy.pi
tt = np.linspace(0, 3, 250)

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={})
for g in [0.1, 0.5, 1.0, 2.0, 5.0]:
    if g == 1:
        x_t = sympy.lambdify(t, x_critical.subs({omega0:w0}), 'numpy')
    else:
        x_t = sympy.lambdify(t, solution.rhs.subs({gamma:g, omega0:w0}), 'numpy')
    ax.plot(tt, x_t(tt), label=(r'$\gamma = %.1f$' %g))
ax.legend()
# %%
