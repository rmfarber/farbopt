TIME=30 #seconds
SIZE=1
rm -f param.dat
./genData.x - $SIZE 0. | ./nloptTrain.x -d - -p param.dat -t $TIME $1
./genData.x - 1 0 | ./pred.x -d - -p param.dat 
