#%%
import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt

#Creating a fuction that returns the solution of the coupled spring system
def coupled_spring_system(t, y, args):
    m1, k1, g1, m2, k2, g2 = args 
    return [y[1], (k2/m1)*(y[2]-y[0]) - (k1/m1)*y[0] - (g1/m1)*y[1], y[3], (-g2/m2)*y[3] - (k2/m2*(y[2]- y[0]))]

# Intial conditions
y0 = [1.0, 0.0, 0.5, 0.0] 
t = np.linspace(0, 20, 1000) # Time array
m1, k1, g1 = 1.0, 10.0, 0.5 # Mass, spring constant, and damping for the first spring
m2, k2, g2 = 2.0, 40.0, 0.25 # Mass, spring constant, and damping for the second spring
args = (m1, k1, g1, m2, k2, g2) # Arguments for the function

# Solving the system of equations using scipy's ode function
sol = integrate.ode(coupled_spring_system)
sol.set_integrator('vode', method='bdf')
sol.set_initial_value(y0, t[0])
sol.set_f_params(args) # Set the parameters for the function

y = np.zeros((len(t), len(y0))) # Create an array to store the solution
dt = t[1] - t[0] # Time step
idx = 0 # Index for the solution array

while sol.successful() and sol.t < t[-1]:
    y[idx, :] = sol.y
    sol.integrate(sol.t + dt) # Integrate the system
    idx += 1 # Increment the index
y_sol = sol.y # Store the final solution
y_solution = interpolate.interp1d(t[:idx, ], y[:idx, ], axis=0) # Interpolate the solution

fig = plt.figure(figsize=(8, 4)) # Create a figure
ax1 = plt.subplot2grid((2, 5), (0, 0), colspan = 3) # Create a subplot for the first spring
ax2 = plt.subplot2grid((2, 5), (1, 0), colspan = 3) # Create a subplot for the second spring
ax3 = plt.subplot2grid((2, 5), (0, 3), colspan = 2, rowspan= 2) # Create a subplot for the first spring velocity

for i in range(0, len(t), 100):
    ax1.plot(t[:i], y[: i, 0], 'r')
    ax2.plot(t[:i], y[:i, 2], 'b')
    ax3.plot(y[:i, 0], y[:i, 2], 'k')
    plt.pause(0.1) # Pause for a short time to create the animation
    plt.savefig('Coupled_Spring_System.png') # Save the figure as a PNG file

plt.show() # Show the plot
# %%
