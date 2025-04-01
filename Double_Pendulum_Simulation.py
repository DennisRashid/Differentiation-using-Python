#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import sympy
sympy.init_printing()

t, g, m1, l1, m2, l2 = sympy.symbols('t, g, m_1, l_1, m_2, l_2')
theta1, theta2 = sympy.symbols('theta_1, theta_2', cls=sympy.Function)

ode1 = sympy.Eq((m1+m2)*l1*theta1(t).diff(t, 2) + m2*l2*theta2(t).diff(t, 2)*sympy.cos((theta1(t)-theta2(t)))+ 
                m2*l2*(theta2(t).diff(t))**2*sympy.sin(theta1(t)-theta2(t)) + g * (m1 + m2)* sympy.sin(theta1(t)), 0)

ode2 = sympy.Eq(m2*l2*theta2(t).diff(t, 2) + m2*l2*theta1(t).diff(t)*sympy.cos(theta1(t)- theta2(t)) -
                m2*l1*(theta1(t).diff(t))**2*sympy.sin(theta1(t)-theta2(t)) + m2*g*sympy.sin(theta2(t)), 0)

y1, y2, y3, y4 = sympy.symbols('y_1, y_2, y_3, y_4', cls=sympy.Function)
varchange = {theta1(t).diff(t, 2): y2(t).diff(t), theta1(t): y1(t), theta2(t).diff(t, 2): y4(t).diff(t), theta2(t):y3(t)}
ode1_vc = ode1.subs(varchange)
ode2_vc = ode2.subs(varchange)

ode3 = sympy.Eq(y1(t).diff(t) - y2(t),0)
ode4 = sympy.Eq(y3(t).diff(t) - y4(t), 0)

y = sympy.Matrix([y1(t), y2(t), y3(t), y4(t)])
vcsol = sympy.solve((ode1_vc, ode2_vc, ode3, ode4), y.diff(t), dict=True)
f = y.diff(t).subs(vcsol[0])
params = {m1:5.0, l1:2.0, m2:1.0, l2:1.0, g:9.81}
f_np = sympy.lambdify((t, y), f.subs(params), 'numpy')
j = sympy.Matrix(f).jacobian(y)
j_np = sympy.lambdify((t, y), j.subs(params), 'numpy')

y0 = [2.0, 0, 0, 0]
t = np.linspace(0, 20, 1000)
r = integrate.ode(f_np, j_np).set_initial_value(y0, t[0])
dt = t[1] - t[0]
y = np.zeros((len(t), len(y0)))
idx = 0
while r.successful()and r.t < t[-1]:
    y[idx, :] = r.y
    r.integrate(r.t + dt)
    idx += 1 

theta1_np, theta2_np = y[:, 0], y[:, 2]
x1 = params[l1] * np.sin(theta1_np)
y1 = -params[l1] * np.cos(theta1_np)
x2 = x1 + params[l2] * np.sin(theta2_np)
y2 = y1 - params[l2] * np.cos(theta2_np)

fig = plt.figure(figsize=(10, 4))
ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=3)
ax2 = plt.subplot2grid((2, 5), (1, 0), colspan=3)
ax3 = plt.subplot2grid((2, 5), (0, 3), colspan=2, rowspan=2)

for i in range(0, len(t), 50):
    ax1.plot(t[:i], x1[:i], 'r')
    ax1.plot(t[:i], y1[:i], 'b')
    ax1.set_ylabel('$x_1, y_1$', fontsize=18)
    ax1.set_yticks([-3, 0, 3])

    ax2.plot(t[:i], x2[:i], 'r')
    ax2.plot(t[:i], y2[:i], 'b')
    ax2.set_xlabel('$t$', fontsize=18)
    ax2.set_ylabel('$x_2, y_2$', fontsize=18)
    ax2.set_yticks([-3, 0, 3])

    ax3.plot(x1[:i], y1[:i], 'r')
    ax3.plot(x2[:i], y2[:i], 'b', lw=0.5)
    ax3.set_xlabel('$x$', fontsize=18)
    ax3.set_ylabel('$y$', fontsize=18)
    ax3.set_xticks([-3, 0, 3])
    ax3.set_yticks([-3, 0, 3])
    plt.pause(0.5)

plt.show()
# %%
