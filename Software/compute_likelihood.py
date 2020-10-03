#! /usr/bin/env python

from emd.emd_normal_lib import *
from treeswift import *
import argparse
from simulator.multinomial import *

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True,help="Input tree")
parser.add_argument("-r","--ref",required=True, help="Reference time tree to compute llh score")
parser.add_argument("-t","--samplingTime",required=True,help="Sampling time")
parser.add_argument("-s","--seqLen",required=False, help="Sequence length. Default: 1000")
parser.add_argument("-k","--nbin",required=False,help="The number of bins to discretize the rate distribution. Default: 100")
#parser.add_argument("-U","--clockFile",required=False, help="A file contains mutation rates to be used to compute emperical rate distribution")
parser.add_argument("--clockFile",required=False,help="A file that defines a customized (discretized) clock model")

args = vars(parser.parse_args())


intreeFile = args["input"]
timeFile = args["samplingTime"]
refTreeFile = args["ref"]
s = int(args["seqLen"]) if args["seqLen"] else 1000
k = int(args["nbin"]) if args["nbin"] else 100
clockFile = args["clockFile"]

smpl_times = {}

with open(timeFile,"r") as fin:
    for line in fin:
        name,time = line.split()
        smpl_times[name] = float(time)

if clockFile:
    omega = []
    phi = []
    with open(clockFile,'r') as f:
        for line in f:
            o,p = line.strip().split()
            omega.append(float(o))
            phi.append(float(p))
    init_rate_distr = multinomial(omega,phi)
    fixed_phi = True
else:
    init_rate_distr = None    
    fixed_phi = False

with open(intreeFile,'r') as fin:
    for treeStr in fin:
        tree = read_tree_newick(treeStr)
        _,_,b,stds = setup_constr(tree,smpl_times,s)
        tau, omega, phi = EM_date(tree,smpl_times,refTreeFile=refTreeFile,maxIter=1,s=s,k=k,fixed_phi=fixed_phi,fixed_tau=True,init_rate_distr=init_rate_distr)        
