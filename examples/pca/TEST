TIME=30 #seconds
SIZE=1
rm -f param.dat
./genData.x - $SIZE 0. | ./nloptTrain.x - param.dat $TIME
./genData.x - 1 0 | ./pred.x - param.dat 
