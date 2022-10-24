#! /usr/bin/env python

from emd.emd_normal_lib import *
from treeswift import *
import argparse
from simulator.multinomial import *
import sys
import time

parser = argparse.ArgumentParser()

parser.add_argument("-i","--input",required=True,help="Input trees")
parser.add_argument("-t","--samplingTime",required=True,help="Sampling time")
parser.add_argument("-k","--ncat",required=False,help="The number of rate categories. Default: 50")
parser.add_argument("-b","--backward",action='store_true',help="Use backward time and enforce ultrametricity. This option is useful for fossil calibrations with present-time sequences. Default: NO") 
parser.add_argument("-o","--output",required=False,help="The output trees with branch lengths in time unit. Default: [input].emDate")
parser.add_argument("-j","--clockout",required=False,help="Write down the estimated clock parameters (omega and phi) to this file. Default: [input].clock")
parser.add_argument("-p","--rep",required=False, help="The number of random replicates for initialization. Default: 100")
parser.add_argument("-l","--seqLen",required=False, help="The length of the sequences. Default: 1000")
parser.add_argument("--clockFile",required=False,help="A file that defines a customized (discretized) clock model. Will override --bins")
parser.add_argument("-v","--verbose",action='store_true',help="Verbose")
parser.add_argument("--maxIter",required=False,help="The maximum number of iterations for EM search. Default: 100")
parser.add_argument("--randSeed",required=False,help="Random seed; either a number or a list of p numbers where p is the number of replicates specified by -p. Default: auto-select")
parser.add_argument("--annotate",required=False,help="Annotation option. Select one of these options: 1: Annotate divergent times; 2: Annotate divergent times and expected mutation rates; 3: Annotate divergent times, expected mutation rates, and the full posterior distribution of the mutation rate. Default: 2")


args = vars(parser.parse_args())

start = time.time()
print("EMDate was called as follow: " + " ".join(sys.argv))

k = int(args["ncat"]) if args["ncat"] else 50
intreeFile = args["input"]
outtreeFile = args["output"] if args["output"] else (intreeFile + ".mdcatTree")
infoFile = args["clockout"] if args["clockout"] else (intreeFile + ".mdcatClock")
nreps = int(args["rep"]) if args["rep"] else 100

timeFile = args["samplingTime"]
seqLen = int(args["seqLen"]) if args["seqLen"] else 1000
smpl_times = {}
maxIter = int(args["maxIter"]) if args["maxIter"] else 100
bw_time = args["backward"]
leaf_time = 0 if bw_time else None

try:
    opt = int(args["annotate"])
except:
    opt = 2
place_mu = (opt >= 2)
place_q = (opt >= 3)

try:
    randseed = [int(x) for x in args["randSeed"].strip().split()]
    if len(randseed) == 1 and nreps != 1:
        randseed = randseed[0]
except:
    randseed = None

tree = read_tree_newick(intreeFile)

best_tree,best_llh,best_phi,best_omega = MDCat(tree,k,sampling_time=timeFile,s=seqLen,nrep=nreps,maxIter=maxIter,refTree=None,fixed_tau=False,fixed_omega=False,verbose=args["verbose"],pseudo=1,randseed=randseed,place_mu=place_mu,place_q=place_q,init_Q=None,root_time=None,leaf_time=leaf_time,bw_time=bw_time)                 
best_tree.write_tree_newick(outtreeFile)
with open(infoFile,'w') as finfo:
    for (o,p) in zip(best_omega,best_phi):
        finfo.write(str(o) + " " + str(p) + "\n")
print("Best log-likelihood: " + str(best_llh))       
end = time.time()
print("Runtime: ", end - start)
