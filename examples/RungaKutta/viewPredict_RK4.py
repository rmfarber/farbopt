# RK2D.py: Plot out time series of integration steps of a 2D ODE
#      to illustrate the fourth-order Runge-Kutta method.
#
# For a 2D ODE
#     dx/dt = f(x,y)
#     dy/dt = g(x,y)
# See RKTwoD() below for how the fourth-order Rungle-Kutta method integrates.
#

# Import plotting routines
from pylab import *
import numpy as np
import struct
from farbopt import PyPredFcn
dt = 0.1

# The van der Pol 2D ODE
def van_der_pol_oscillator_deriv(x, t):
    nx0 = x[1]
    nx1 = -mu * (x[0] ** 2.0 - delta) * x[1] - omega * x[0]
    res = np.array([nx0, nx1])
    return res

def predicted_rhs(x,t):
    return RHS.predict(x)

# 2D Fourth-Order Runge-Kutta Integrator
def RKTwoD(x, f, t):
    x = np.array(x)
    k1 = 0.1 * f(x,t)
    k2 = 0.1 * f(x + x / 2.0,t)
    k3 = 0.1 * f(x + k2 / 2.0,t)
    k4 = 0.1 * f(x + k3,t)
    x = x + ( k1 + 2.0 * k2 + 2.0 * k3 + k4 ) / 6.0
    return x

# Simulation parameters
# Integration time step

# Time
t  = [ 0.0]
# The number of time steps to integrate over
nSamples = 10000
#mu = 1.
#delta = 4.0
#omega = 1.0

mu = 0.2
delta = 1.0
omega = 1.0

# The main loop that generates the orbit, storing the states
xs= np.empty((nSamples+1,2))
xs[0]=[0.2,0.2]

RHS=PyPredFcn(b'param.dat')
for i in range(0,nSamples):
  # at each time step calculate new x(t) and y(t)
  xs[i+1]=(RKTwoD(xs[i],predicted_rhs,dt))
  t.append(t[i] + dt)
    
#for i in range(0,nSamples):
 #   print(xs[i],xs[i+1])


# Setup the parametric plot
xlabel('x(t)') # set x-axis label
ylabel('y(t)') # set y-axis label
title('4th order Runge-Kutta Method: van der Pol ODE at u = ' + str(mu)) # set plot title
#axis('equal')

# Plot the trajectory in the phase plane
plot(xs[:,0],xs[:,1],'b')
show()

