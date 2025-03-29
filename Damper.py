#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy
sympy.init_printing()

t, omega0, gamma = sympy.symbols('t, omega_0, gamma', positive=True)
X = sympy.Function('x')

ode = sympy.Eq(X(t).diff(t, 2) + (2* gamma * omega0 * X(t).diff(t)) + ((omega0**2 *X(t))), 0)
ode_soln = sympy.dsolve(ode)

ics = {X(0): 1, X(t).diff(t).subs(t, 0): 0}
def apply_ics(sol, ics, x, known_params):
    free_params = sol.free_symbols - set(known_params) 
    eq = [((sol.lhs.diff(x, n)-sol.rhs.diff(x, n)).subs(x, 0).subs(ics)) for n in range(len(ics))]
    C_solns = sympy.solve(eq, free_params)
    return sol.subs(C_solns)

solution = apply_ics(ode_soln, ics, t, [omega0, gamma])
X_critical = sympy.limit(solution.rhs, gamma, 1)

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={})
w0 = 2 *sympy.pi
tt = np.linspace(0, 3, 250)
for g in [0.1, 0.5, 1.0, 2.0, 5.0]:
    if g == 1:
        x_t = sympy.lambdify(t, X_critical.subs({omega0:w0}), 'numpy')

    else:
        x_t = sympy.lambdify(t, solution.rhs.subs({gamma:g, omega0:w0}),  'numpy')

    ax.plot(tt, x_t(tt).real, label=r"$\gamma = %.1f$" % g)
ax.set_xlim(tt.min(), tt.max())
ax.set_ylim(-0.85, 1.1)
ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel(r"$t$", fontsize=18)
ax.set_ylabel(r"$X(t)$", fontsize=18)
ax.legend()
ax.set_title('Damper Harmonic Oscillator', fontsize=18)
plt.show()
# %%
