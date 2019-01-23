build the codes
# generate the data
sh HOST_GEN_TRAIN 

#train the network (example uses 10 seconds, try longer for better accuracy e.g. 3600)
./nloptTrain.x --data train.dat --param param.dat -t 10

#predict using trained network, use NRMSD as measure
./pred.x -d pred.dat -p param.dat | ./NRMSE.py

