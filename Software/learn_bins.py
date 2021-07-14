#! /usr/bin/env python

from simulator.multinomial import multinomial
from treeswift import *
from emd.binning_lib import init_bins, refine_bins, make_bins
from emd.emd_normal_lib import *
from sys import argv

def main():  
    mu = 0.006
    k = 100
    treefile = argv[1]
    timefile = argv[2]
    trueTreeFile = argv[3]
    rateFile = argv[4] # output

    smplTime = {}
    with open(timefile,'r') as f:
        for line in f:
            s,t = line.strip().split()
            smplTime[s] = float(t)

    omega,phi = init_bins(mu,k)    
    
    tree = read_tree_newick(treefile)
    rate_distr = multinomial(omega,phi)
    _,omega,phi = EM_date(tree,smplTime,refTreeFile=trueTreeFile,df=0.001,s=1000000,fixed_phi=False,fixed_tau=True,init_rate_distr=rate_distr)  
    
    with open(rateFile,'w') as fout:
        for (o,p) in zip(omega,phi):
            fout.write(str(o) + " " + str(p) + "\n")

if __name__ == "__main__":
    main()
