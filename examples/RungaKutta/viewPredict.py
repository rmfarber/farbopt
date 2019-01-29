import numpy as np
from pylab import *
from farbopt import PyPredFcn

from scipy.integrate import odeint

#Settings used in https://arxiv.org/pdf/comp-gas/9305001.pdf
nSamples=5000
mu = 1
delta = 4.0
omega = 1.0

#mu = 0.2
#delta = 1.0
#omega = 1.0

def predicted_rhs(x,t):
    return RHS.predict(x)
    
def van_der_pol_oscillator_deriv(x, t):
    nx0 = x[1]
    nx1 = -mu * (x[0] ** 2.0 - delta) * x[1] - omega * x[0]
    res = np.array([nx0, nx1])
    return res

ts = np.linspace(0.0, 50.0, nSamples)
RHS=PyPredFcn(b'param.dat')

xs = odeint(predicted_rhs, [0.2, 0.2], ts)
plt.plot(xs[:,0], xs[:,1])
#xs = odeint(van_der_pol_oscillator_deriv, [-3.0, -3.0], ts)
##plt.plot(xs[:,0], xs[:,1])
xs = odeint(van_der_pol_oscillator_deriv, [4.0, 4.0], ts)
#plt.plot(xs[:,0], xs[:,1])
#plt.gca().set_aspect('equal')
#plt.savefig('vanderpol_oscillator.png')
plt.show() 
