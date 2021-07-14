#! /usr/bin/env python

#from emd.emd_lib import *
from emd.emd_normal_lib import *
from treeswift import *
import argparse
from simulator.multinomial import *
from random import random
from emd.binning_lib import init_bins

parser = argparse.ArgumentParser()

parser.add_argument("-i","--input",required=True,help="Input trees")
parser.add_argument("-t","--samplingTime",required=True,help="Sampling time")
parser.add_argument("-o","--output",required=False,help="The output trees with branch lengths in time unit. Default: [input].emDate")
parser.add_argument("-j","--estParam",required=False,help="Write down the estimated parameters (omega and phi) to this file. Default: [input].emParam")
parser.add_argument("-p","--rep",required=False, help="The number of random replicates for initialization. Default: 100")
parser.add_argument("-l","--seqLen",required=False, help="The length of the sequences. Default: 10000")
parser.add_argument("--assignLabel",action='store_true',help="Assign label to internal nodes. Default: NO")
parser.add_argument("-v","--verbose",action='store_true',help="Verbose")
parser.add_argument("-k","--nbin",required=False,help="The maximum number of bins to discretize the rate distribution. Default: 130")
parser.add_argument("--maxIter",required=False,help="The maximum number of iterations for EM search. Default: 100")
parser.add_argument("--mu",required=True,help="The global mutation rate")

args = vars(parser.parse_args())

intreeFile = args["input"]
outtreeFile = args["output"] if args["output"] else (intreeFile + ".emDate")
infoFile = args["estParam"] if args["estParam"] else (intreeFile + ".emParam")
nreps = int(args["rep"]) if args["rep"] else 100

timeFile = args["samplingTime"]
seqLen = int(args["seqLen"]) if args["seqLen"] else 10000
k = int(args["nbin"]) if args["nbin"] else 130
smpl_times = {}
maxIter = int(args["maxIter"]) if args["maxIter"] else 100
mu = float(args["mu"])

with open(timeFile,"r") as fin:
    for line in fin:
        name,time = line.split()
        smpl_times[name] = float(time)

tree = read_tree_newick(intreeFile)
if args["assignLabel"]:
    nodeIdx = 0
    for node in tree.traverse_preorder():
        if not node.is_leaf():
            node.set_label("I" + str(nodeIdx))
            nodeIdx += 1    
                   
best_tree,best_llh,best_phi,best_omega = EM_date_adapt_bins(tree,smpl_times,mu,nbins=k,s=seqLen,maxIter=maxIter,nrep=nreps)
best_tree.write_tree_newick(outtreeFile)
with open(infoFile,'w') as finfo:
    for (o,p) in zip(best_omega,best_phi):
        finfo.write(str(o) + " " + str(p) + "\n")
print("Best likelihood: " + str(best_llh))        
