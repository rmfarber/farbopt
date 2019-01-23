#!/bin/bash
. ./common.sh


if [ $DIM -ne "2" ];then
   echo "wrong input vector length. Must be 2";
   exit -1;
fi

TYPE=`basename $PWD`
KNOWN_PNG=$TYPE"_known.png"
#echo $KNOWN_PNG
PRED_PNG=$TYPE"_pred.png"
#echo $PRED_PNG

sed '1,4d' pred.csv > plot.txt
gnuplot -e  "unset key; set term png; set output \""$PRED_PNG"\"; plot \"plot.txt\" u 2:3"
gnuplot -e  "unset key; set term png; set output \""$KNOWN_PNG"\"; plot \"plot.txt\" u 5:6"
