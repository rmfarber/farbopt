This generates data and trains an autoencoder to solve an example problem. 

The default input vector length is 2.

NOTE: You need to add the adolc library to your path to run:
source ../ADD_ADOLC

You only need to do this once

To get performance numbers type:
make
sh RUN.sh


You can change the length of the input vector with CHANGE_INPUTSIZE.sh.
For example set DIM=16 to train using 16 floats per input vector.

   sh CHANGE_INPUTSIZE.sh 16
   make
   sh RUN.sh


