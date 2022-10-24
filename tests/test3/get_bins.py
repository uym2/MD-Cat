#! /usr/bin/env python

from sys import argv

k=int(argv[1])
maxBin = float(argv[2])

for i in range(k):
    print(str(maxBin*(2*i+1)/2/k) + " " + str(1/k))
