This directory demonstrates how an ANN can train itself to learn the ODE
that can be used in an integrator. In this case the ANN learns to become
the Van Der Pol oscillator.

Please look to the following paper to get the details of this experiment:
https://arxiv.org/pdf/comp-gas/9305001.pdf

There are defines in FcnOfInterest.hpp to implement the explicit and implicit
versions.

Look to the variable PREDFCN to see how to directly call the rhs() function in
FcnOfInterest.hpp.

This also builds a Python callable method as shown in viewPredict.py

The python method genVanDerPol.py generates the train.dat data.

The viewPredict.py uses the python method to use the rhs() in the standard
Python odeint() method. You can see the stable limit cycle on the screen.

To use:

make -j
python genVanDerPol.py
./nloptTrain.x -p param.dat -d train.dat -t 600
sh BUILD.python.module.sh
python viewPredict.py
