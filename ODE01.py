#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import sympy
sympy.init_printing()

t, k, T0, Ta = sympy.symbols('t, k, T_0, T_a')
T = sympy.Function('T')
ode = sympy.Eq(sympy.diff(T(t), t) + k*(T(t)-Ta), 0)
ode_soln = sympy.dsolve(ode)

ics = {T(0):T0}
C_eq = sympy.Eq(ode_soln.lhs.subs(t, 0).subs(ics), ode_soln.rhs.subs(t, 0))
C_soln = sympy.solve(C_eq)
ode_Tt_soln = ode_soln.subs(C_soln[0])

T_Ambient = 25
T_Initial = 37
Time_Taken = 5
Conductivity = 0.1

Body_Temp = ode_Tt_soln.subs({t:Time_Taken, T0: T_Initial, Ta: T_Ambient, k:Conductivity})
# %%
