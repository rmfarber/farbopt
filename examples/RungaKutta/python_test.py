# Python running example

from __future__ import print_function

import numpy as np
from farbopt import PyPredFcn

# Initialising the wrapped c++ function
R1 = PyPredFcn(b'param.dat');

list6 = np.array([1, 2])
print("Test return, Sum list: ", R1.predict(list6))

# This should work with any n-dimensional array
