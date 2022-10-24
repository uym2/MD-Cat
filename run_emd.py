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
parser.add_argument("-b","--backward",action='store_true',help="Use backward time and enforce ultrametricity. This option is useful for fossil calibrations with present-time sequences. Default: NO") 
parser.add_argument("-o","--output",required=False,help="The output trees with branch lengths in time unit. Default: [input].emDate")
parser.add_argument("-j","--estParam",required=False,help="Write down the estimated parameters (omega and phi) to this file. Default: [input].emParam")
parser.add_argument("-p","--rep",required=False, help="The number of random replicates for initialization. Default: 100")
parser.add_argument("-l","--seqLen",required=False, help="The length of the sequences. Default: 1000")
parser.add_argument("-f","--refTreeFile",required=False, help="A reference time tree as initial solution. Default: None")
parser.add_argument("--assignLabel",action='store_true',help="Assign label to internal nodes. Default: NO")
parser.add_argument("--clockFile",required=False,help="A file that defines a customized (discretized) clock model. Will override --bins")
parser.add_argument("--muAvg",required=False,help="Fix the average mutation rate to this value in the first round search")
parser.add_argument("--bins",required=False,help="Specify the bins for the rate (i.e. omega)")
parser.add_argument("--fixedTau",action='store_true',help="Fix the time tree.")
parser.add_argument("--fixedOmega",action='store_true',help="Fix the rate categories.")
parser.add_argument("-v","--verbose",action='store_true',help="Verbose")
parser.add_argument("--maxIter",required=False,help="The maximum number of iterations for EM search. Default: 100")
parser.add_argument("--pseudo",required=False,help="Adding pseudo-count to each branch length (divided by sequence length). Default: 0")
parser.add_argument("--randSeed",required=False,help="Random seed; either a number or a list of p numbers where p is the number of replicates specified by -p. Default: auto-select")
parser.add_argument("--annotate",required=False,help="Annotation option. Select one of these options: 1: Annotate divergent times; 2: Annotate divergent times and expected mutation rates; 3: Annotate divergent times, expected mutation rates, and the full posterior distribution of the mutation rate. Default: 2")


args = vars(parser.parse_args())

start = time.time()
print("EMDate was called as follow: " + " ".join(sys.argv))

intreeFile = args["input"]
outtreeFile = args["output"] if args["output"] else (intreeFile + ".emDate")
infoFile = args["estParam"] if args["estParam"] else (intreeFile + ".emParam")
nreps = int(args["rep"]) if args["rep"] else 100

timeFile = args["samplingTime"]
seqLen = int(args["seqLen"]) if args["seqLen"] else 1000
refTreeFile = args["refTreeFile"]
smpl_times = {}
maxIter = int(args["maxIter"]) if args["maxIter"] else 100
fixedTau = args["fixedTau"]
fixedOmega = args["fixedOmega"]
muAvg = float(args["muAvg"]) if args["muAvg"] is not None else None
bw_time = args["backward"]
leaf_time = 0 if bw_time else None

try:
    opt = int(args["annotate"])
except:
    opt = 2
place_mu = (opt >= 2)
place_q = (opt >= 3)

refTree = read_tree_newick(refTreeFile) if refTreeFile else None
try:
    randseed = [int(x) for x in args["randSeed"].strip().split()]
    if len(randseed) == 1 and nreps != 1:
        randseed = randseed[0]
except:
    randseed = None

pseudo = float(args["pseudo"]) if args["pseudo"] else 0

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
    k = len(omega)
    phi = [1/k]*k
    init_rate_distr = multinomial(omega,phi)
else:
    print("ERROR: neither --clockFile or --bins is specified. Please specify one of those two options. Exit now!")
    exit()

tree = read_tree_newick(intreeFile)
if args["assignLabel"]:
    nodeIdx = 0
    for node in tree.traverse_preorder():
        if not node.is_leaf():
            node.set_label("I" + str(nodeIdx))
            nodeIdx += 1           

best_tree,best_llh,best_phi,best_omega = MDCat(tree,init_rate_distr,sampling_time=timeFile,s=seqLen,nrep=nreps,maxIter=maxIter,refTree=refTree,fixed_tau=fixedTau,fixed_omega=fixedOmega,verbose=args["verbose"],mu_avg=muAvg,pseudo=pseudo,randseed=randseed,place_mu=place_mu,place_q=place_q,init_Q=init_Q,root_time=None,leaf_time=leaf_time,bw_time=bw_time)                 

best_tree.write_tree_newick(outtreeFile)
with open(infoFile,'w') as finfo:
    for (o,p) in zip(best_omega,best_phi):
        finfo.write(str(o) + " " + str(p) + "\n")
print("Best log-likelihood: " + str(best_llh))       
end = time.time()
print("Runtime: ", end - start)
