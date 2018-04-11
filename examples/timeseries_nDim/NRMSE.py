#!/usr/bin/python

import sys, fileinput, re
from math import sqrt

#skip header info
for i in range(0,8):
    sys.stdin.readline()

lines=0
total=0.
min_p=0.
max_p=0.
for line in fileinput.input():
        lines += 1
        x=line.split(', ')
        pred = float(x[1])
        known = float(x[3])

	if lines == 0:
	   min_p = pred
	   max_p = pred

	if min_p > pred:
	   min_p = pred
	if max_p < pred:
	   max_p = pred

        total += (pred-known) * (pred-known)
        

print "NRMSD %.4f" % (sqrt(total/lines)/(max_p-min_p))
