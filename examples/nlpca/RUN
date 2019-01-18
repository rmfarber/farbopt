TIME=30 #seconds
SIZE=60000
NDIM=2
NPRED=100
SEED1=1234
SEED2=123
rm -f train.dat param.dat
./genData.x - $SIZE 0.01 $SEED1 $NDIM | ./nloptTrain.x - param.dat $TIME
./genData.x - $NPRED 0 $SEED2 $NDIM | ./pred.x - param.dat  > pred.csv
