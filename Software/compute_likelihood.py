#! /usr/bin/env python

from emd.emd_lib import *
from treeswift import *
import argparse
from simulator.multinomial import *

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True,help="Input tree")
parser.add_argument("-r","--ref",required=True, help="Reference time tree to compute llh score")
parser.add_argument("-t","--samplingTime",required=True,help="Sampling time")
parser.add_argument("-s","--seqLen",required=False, help="Sequence length. Default: 1000")
parser.add_argument("-k","--nbin",required=False,help="The number of bins to discretize the rate distribution. Default: 100")
parser.add_argument("-U","--muFile",required=False, help="A file contains mutation rates to be used to compute emperical rate distribution")

args = vars(parser.parse_args())


intreeFile = args["input"]
timeFile = args["samplingTime"]
refTreeFile = args["ref"]
s = int(args["seqLen"]) if args["seqLen"] else 1000
k = int(args["nbin"]) if args["nbin"] else 100
muFile = args["muFile"]

smpl_times = {}

with open(timeFile,"r") as fin:
    fin.readline()
    for line in fin:
        name,time = line.split()
        smpl_times[name] = float(time)

if muFile:
    mus = []
    with open(muFile,'r') as f:
        for line in f:
            mus.append(float(line))
    #init_rate_distr = emperical_histogram(mus,k)      
    omega = mus
    k = len(omega)
    phi = [1/k]*k  
    init_rate_distr = multinomial(omega,phi)
else:
    init_rate_distr = None    

tree = read_tree_newick(intreeFile)
M, dt, x = setup_constr(tree,smpl_times,s)

with open(refTreeFile,'r') as fin:
    for line in fin:
        fout = open("temp","w")
        fout.write(line)
        fout.close()
        #tau, omega, phi = init_EM(tree,smpl_times,k,s=s,refTreeFile="temp")
        tau, omega, phi = EM_date(tree,smpl_times,refTreeFile="temp",s=s,k=k,fixed_phi=True,fixed_tau=True,init_rate_distr=init_rate_distr)        
        llh = f_ll(x,s,tau,omega,phi)
        print("Likelihood: " + str(llh))
