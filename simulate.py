#! /usr/bin/env python

from simulator.multinomial import *
from simulator.sim_lib import *
from treeswift import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True,help="Input tree; branch lengths must be in time unit")
parser.add_argument("-o","--output",required=True,help="Output tree(s)")
parser.add_argument("-m","--model",required=False, help="Model of the rate distribution. Either lnorm_[sd], exp, gamma_[sd], or const. Default: const")
parser.add_argument("-u","--mu",required=False, help="Expected mutation rate. Default: 1.0")
parser.add_argument("-s","--seqLen",required=False, help="Sequence length. Needed for the Gaussian branch error (see -b). Default: 1000")
parser.add_argument("-n","--nsample",required=False,help="Number of samples to be generated. Default: 1")
parser.add_argument("-k","--nbin",required=False,help="The number of bins to discretize the rate distribution. Default: Do not discretize (i.e. k = inf)")
parser.add_argument("-f","--inMuFile",required=False,help="The file that defines a customized categorical clock model. Will override -u, -k, and -m")
parser.add_argument("-U","--outMuFile",required=False,help="Write the simulated mu values to this file. Default: None")
parser.add_argument("-b","--brError",required=False,help="Model of the branch error. Can be either Gaussian (lsd's model), Poisson (LF's model), or None (no branch error). Default: None")
parser.add_argument("--multimodal",required=False,help="Define a multimodal distribution. Will override -m, -k, and -u")

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

model_args = model.split("_") if model else [None,None]

if args["multimodal"] is not None:    
    models = []
    probs = []
    for x in args["multimodal"].split():
        a,b,c = x.split(":")
        m = a.split("_")
        mu = float(b)
        p = float(c)
        if m[0] == 'lnorm':
            sd = float(m[1])
            models.append(lognormal(mu,sd))
        elif m[0] == 'gamma':
            sd = float(m[1])
            models.append(gamma(mu,sd))
        elif m[0] == 'exp':
            models.append(exponential(mu))
        probs.append(p)
    rate_distr = multimodal(models,probs)                
elif args["inMuFile"] is not None:
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
elif model_args[0] == 'gamma':
    sd = float(model_args[1])
    rate_distr = discrete_gamma(mu,sd,k) if k is not None else gamma(mu,sd)
elif model_args[0] == 'exp':
    rate_distr = discrete_exponential(mu,k) if k is not None else exponential(mu)
elif model_args[0] == "unif":
    rate_distr = discrete_uniform(mu,k) if k is not None else uniform(mu)    
else:
    rate_distr = None    

tree = read_tree_newick(intreeFile)
nameMap = {"Gaussian":simulate_gaussian,"Poisson":simulate_poisson,"Const":simulate_scale}
f_sim = nameMap[brError]

for i in range(n):
    mus = f_sim(tree,rate_distr=rate_distr,seqLen=s)
    write_tree(tree,edge_type='b',outfile=outtreeFile,append=True)
    if outMuFile:
        with open(outMuFile,'a') as fout:
            for lb in sorted(mus):
                fout.write(lb + " " + str(mus[lb]) + "\n")
