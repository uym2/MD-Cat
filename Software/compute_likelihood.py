#! /usr/bin/env python

from emd.emd_lib import *
from treeswift import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True,help="Input tree")
parser.add_argument("-r","--ref",required=True, help="Reference time tree to compute llh score")
parser.add_argument("-t","--samplingTime",required=True,help="Sampling time")

args = vars(parser.parse_args())


intreeFile = args["input"]
timeFile = args["samplingTime"]
refTreeFile = args["ref"]


smpl_times = {}
s=1000
k=100

with open(timeFile,"r") as fin:
    fin.readline()
    for line in fin:
        name,time = line.split()
        smpl_times[name] = float(time)

tree = read_tree_newick(intreeFile)
M, dt, x = setup_constr(tree,smpl_times,s)
tau, phi, omega = init_EM(tree,smpl_times,k,s=s,refTreeFile=refTreeFile)
llh = f_ll(x,s,tau,omega,phi)

print("Likelihood: " + str(llh))
