#! /usr/bin/env python

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
parser.add_argument("-l","--seqLen",required=False, help="The length of the sequences. Default: 1000")
parser.add_argument("-f","--refTreeFile",required=False, help="A reference time tree as initial solution. Default: None. LSD will be run internally and used as reference")
parser.add_argument("--assignLabel",action='store_true',help="Assign label to internal nodes. Default: NO")
parser.add_argument("--clockFile",required=False,help="A file that defines a customized (discretized) clock model. Will override --bins")
parser.add_argument("--bins",required=False,help="Specify the bins for the rate (i.e. omega)")
parser.add_argument("--fixedPhi",action='store_true',help="Fix the probability distribution and optimize the bin positions instead.")
parser.add_argument("-v","--verbose",action='store_true',help="Verbose")
parser.add_argument("-k","--nbin",required=False,help="The number of bins to discretize the rate distribution. Default: 100")
parser.add_argument("--maxIter",required=False,help="The maximum number of iterations for EM search. Default: 100")
parser.add_argument("--extraData",required=False,help="The extra observations per branch. Default: None")


args = vars(parser.parse_args())

intreeFile = args["input"]
outtreeFile = args["output"] if args["output"] else (intreeFile + ".emDate")
infoFile = args["estParam"] if args["estParam"] else (intreeFile + ".emParam")
nreps = int(args["rep"]) if args["rep"] else 100

timeFile = args["samplingTime"]
seqLen = int(args["seqLen"]) if args["seqLen"] else 1000
k = int(args["nbin"]) if args["nbin"] else 100
refTreeFile = args["refTreeFile"]
smpl_times = {}
maxIter = int(args["maxIter"]) if args["maxIter"] else 100
fixedPhi = args["fixedPhi"]

refTree = read_tree_newick(refTreeFile) if refTreeFile else None

extraData = {}
if args["extraData"] is not None:
    with open(args["extraData"],'r') as fin:
        for line in fin:
            name,length = line.split()
            if name in extraData:
                extraData[name].append(float(length))
            else:    
                extraData[name] = [float(length)]

with open(timeFile,"r") as fin:
    for line in fin:
        name,time = line.split()
        smpl_times[name] = float(time)

omega = None
init_rate_distr = None
if args["clockFile"] is not None:
    omega = []
    phi = []
    with open(args["clockFile"]) as fin:
        for line in fin:
            o,p = line.strip().split()
            omega.append(float(o))
            phi.append(float(p))
    sp = sum(phi)
    phi = [p/sp for p in phi]        
    init_rate_distr = multinomial(omega,phi)
elif args["bins"] is not None:        
    omega = [float(o) for o in args["bins"].split()]
else:
    mu = 0.006 # hard code for now
    omega,phi = init_bins(mu,k)
    #init_rate_distr = multinomial(omega,phi)        
    init_rate_distr = None


tree = read_tree_newick(intreeFile)
if args["assignLabel"]:
    nodeIdx = 0
    for node in tree.traverse_preorder():
        if not node.is_leaf():
            node.set_label("I" + str(nodeIdx))
            nodeIdx += 1           

best_tree,best_llh,best_phi,best_omega = EM_date_random_init(tree,smpl_times,input_omega=omega,init_rate_distr=init_rate_distr,s=seqLen,nrep=nreps,maxIter=maxIter,refTree=refTree,fixed_phi=fixedPhi,fixed_tau=False,k=k,verbose=args["verbose"],extra_data=extraData)                 
best_tree.write_tree_newick(outtreeFile)
with open(infoFile,'w') as finfo:
    for (o,p) in zip(best_omega,best_phi):
        finfo.write(str(o) + " " + str(p) + "\n")
print("Best likelihood: " + str(best_llh))        
