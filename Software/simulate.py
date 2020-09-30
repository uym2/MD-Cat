#! /usr/bin/env python

from simulator.multinomial import *
from simulator.sim_lib import *
from treeswift import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True,help="Input tree; branch lengths must be in time unit")
parser.add_argument("-o","--output",required=True,help="Output tree(s)")
parser.add_argument("-m","--model",required=False, help="Model of the rate distribution. Either lnorm_[sd], exp, or const. Default: const")
parser.add_argument("-u","--mu",required=False, help="Expected mutation rate. Default: 1.0")
parser.add_argument("-s","--seqLen",required=False, help="Sequence length. Default: 1000")
parser.add_argument("-n","--nsample",required=False,help="Number of samples to be generated. Default: 1")
parser.add_argument("-k","--nbin",required=False,help="The number of bins to discretize the rate distribution. Default: Do not discretize (i.e. k = inf)")
parser.add_argument("-f","--inMuFile",required=False,help="The file that defines a customized categorical clock model. Will override -u, -k, and -m")
parser.add_argument("-U","--outMuFile",required=False,help="Write the simulated mu values to this file. Default: None")
parser.add_argument("-b","--brError",required=False,help="Model of the branch error. Can be either Gaussian (lsd's model), Poisson (LF's model), or None (no branch error). Default: None")

args = vars(parser.parse_args())


intreeFile = args["input"]
outtreeFile = args["output"]
model = args["model"]
mu = float(args["mu"]) if args["mu"] else 1.0
s = int(args["seqLen"]) if args["seqLen"] else 1000
n = int(args["nsample"]) if args["nsample"] else 1
k = int(args["nbin"]) if args["nbin"] else None
outMuFile = args["outMuFile"]
brError = args["brError"] if args["brError"] else "Const"
#do_poisson = args["poisson"]

model_args = model.split("_") if model else [None,None]

if args["inMuFile"] is not None:
    omega = []
    phi = []
    with open(args["inMuFile"],'r') as fin:
        for line in fin:
            o,p = line.strip().split()
            omega.append(float(o))
            phi.append(float(p))
    rate_distr = multinomial(omega,phi)        
elif model_args[0] == 'lnorm':
    sd = float(model_args[1])
    rate_distr = discrete_lognorm(mu,sd,k) if k is not None else lognormal(mu,sd)
elif model_args[0] == 'exp':
    rate_distr = discrete_exponential(mu,k) if k is not None else exponential(mu)
else:
    rate_distr = None    

#omega = [0.001,0.01]
#phi = [0.5,0.5]
#rate_distr = multinomial(omega,phi)

tree = read_tree_newick(intreeFile)
nameMap = {"Gaussian":simulate_gaussian,"Poisson":simulate_poisson,"Const":simulate_scale}
f_sim = nameMap[brError]

for i in range(n):
    mus = f_sim(tree,rate_distr=rate_distr,seqLen=s)
    write_tree(tree,edge_type='b',outfile=outtreeFile,append=True)
    if outMuFile:
        with open(outMuFile,'a') as fout:
            fout.write("Tree " + str(i+1) + "\n")
            for lb in mus:
                fout.write(lb + " " + str(mus[lb]) + "\n")
