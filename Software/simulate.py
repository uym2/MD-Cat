#! /usr/bin/env python

from simulator.multinomial import *
from simulator.sim_lib import *
from treeswift import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True,help="Input tree; branch lengths must be in time unit")
parser.add_argument("-o","--output",required=True,help="Output tree(s)")
parser.add_argument("-m","--model",required=True, help="Model of the rate distribution. Either lnorm_[sd] or exp")
parser.add_argument("-u","--mu",required=True, help="Expected mutation rate")
parser.add_argument("-s","--seqLen",required=False, help="Sequence length. Default: 1000")
parser.add_argument("-n","--nsample",required=False,help="Number of samples to be generated. Default: 1")
parser.add_argument("-k","--nbin",required=False,help="The number of bins to discretize the rate distribution. Default: Do not discretize (i.e. k = inf)")
parser.add_argument("-U","--outMuFile",required=False,help="Write the simulated mu values to this file. Default: None")


args = vars(parser.parse_args())


intreeFile = args["input"]
outtreeFile = args["output"]
model = args["model"]
mu = float(args["mu"])
s = int(args["seqLen"]) if args["seqLen"] else 1000
n = int(args["nsample"]) if args["nsample"] else 1
k = int(args["nbin"]) if args["nbin"] else None
outMuFile = args["outMuFile"]

model_args = model.split("_")
if model_args[0] == 'lnorm':
    sd = float(model_args[1])
    rate_distr = discrete_lognorm(mu,sd,k) if k is not None else lognormal(mu,sd)
else:
    rate_distr = discrete_exponential(mu,k) if k is not None else exponential(mu)

#omega = [0.001,0.01]
#phi = [0.5,0.5]
#rate_distr = multinomial(omega,phi)

tree = read_tree_newick(intreeFile)

for i in range(n):
    mus = simulate_poisson(tree,rate_distr,s)
    write_tree(tree,edge_type='b',outfile=outtreeFile,append=True)
    if outMuFile:
        with open(outMuFile,'w') as fout:
            for mu in mus:
                fout.write(str(mu) + "\n")

