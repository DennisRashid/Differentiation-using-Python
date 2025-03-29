#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import sympy
sympy.init_printing()

T0, Ta, t, k = sympy.symbols('T_0, T_a, t, k')
T = sympy.Function('T')

ode = sympy.Eq(T(t).diff(t), -k*(T(t)-Ta))
ode_soln = sympy.dsolve(ode)

ics = {T(0): T0}
def apply_ics(sol, ics, x, known_params):
    free_params = sol.free_symbols - set(known_params)
    eq = [((sol.lhs.diff(x, n)-sol.rhs.diff(x, n)).subs(x, 0).subs(ics)) for n in range(len(ics))]
    sol_params = sympy.solve(eq, free_params)
    return sol.subs(sol_params)

solution = apply_ics(ode_soln, ics, t, [k, Ta])

T_ambient = 25
T_Body = 37
Time_Take = 5
Conductivity = 0.1

Body_Temp = solution.subs({T0:T_Body, Ta:T_ambient, k:Conductivity, t:Time_Take})
# %%
