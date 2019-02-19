This directory demonstrates how an ANN can train an LSTM echo network.

Basically the LSTM is presented with a sequence that it must "remember"
after which is is required to echo the sequence. This ensures that
the LSTM memory is working.

NOTE: You need to add the adolc library to your path to run:

source ../ADD_ADOLC

You only need to do this once.

To use:

make -j
./genEchoRandom.x train.dat 10000 1234
./genEchoRandom.x pred.dat 20 2345
rm -fr param.dat
./nloptTrain.x -p param.dat -d train.dat -t 120
sh BUILD.python.module.sh
./pred.x -p param.dat -d pred.dat 

