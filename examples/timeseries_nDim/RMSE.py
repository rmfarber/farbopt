#!/usr/bin/python

import sys, fileinput, re
from math import sqrt

#skip header info
for i in range(0,8):
    sys.stdin.readline()

lines=0
total=0.
for line in fileinput.input():
        lines += 1
        x=line.split(', ')
        pred = float(x[1])
        known = float(x[3])
        total += (pred-known) * (pred-known)
        

print "RMSD %.4f" % sqrt(total/lines)
