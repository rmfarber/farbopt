. ./common.sh
TIME=30 #seconds
SIZE=6000
NPRED=100
SEED1=1234
SEED2=123
rm -f train.dat param.dat
./genData.x - $SIZE 0.01 $SEED1 $DIM | ./nloptTrain.x -d - -p param.dat -t $TIME
./genData.x - $NPRED 0 $SEED2 $DIM | ./pred.x - param.dat  > pred.csv
