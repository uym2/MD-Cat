#! /usr/bin/env python

#from emd.emd_lib import *
from emd.emd_normal_lib import *
from treeswift import *
import argparse
from simulator.multinomial import *

parser = argparse.ArgumentParser()

parser.add_argument("-i","--input",required=True,help="Input trees")
parser.add_argument("-t","--samplingTime",required=True,help="Sampling time")
parser.add_argument("-r","--rootAge",required=False,help="Root age")
parser.add_argument("-o","--output",required=False,help="The output trees with branch lengths in time unit. Default: [input].emDate")
parser.add_argument("-j","--estParam",required=False,help="Write down the estimated parameters (omega and phi) to this file. Default: [input].emParam")
parser.add_argument("-p","--rep",required=False, help="The number of random replicates for initialization. Default: use lsd initialization instead")
parser.add_argument("-l","--seqLen",required=False, help="The length of the sequences. Default: 1000")
parser.add_argument("-f","--refTreeFile",required=False, help="A reference time tree as initial solution. Default: None. LSD will be run internally and used as reference")
parser.add_argument("-k","--nbin",required=False,help="The number of bins to discretize the rate distribution. Default: 100")
parser.add_argument("--trueTreeFile",required=False,help="For debugging purposes. The true tree in substitutions unit; would be used to compute the variance of the Gaussian model. Default: None")
parser.add_argument("-u","--pseudo",required=False,help="Pseudo count. Default: 0 if trueTreeFile is specified else 1")
parser.add_argument("--assignLabel",action='store_true',help="Assign label to internal nodes. Default: NO")
parser.add_argument("--clockFile",required=False,help="A file that defines a customized (discretized) clock model")
parser.add_argument("--fixedPhi",action='store_true',help="Fix the probability distribution")

args = vars(parser.parse_args())

intreeFile = args["input"]
outtreeFile = args["output"] if args["output"] else (intreeFile + ".emDate")
infoFile = args["estParam"] if args["estParam"] else (intreeFile + ".emParam")

timeFile = args["samplingTime"]
rootAge = float(args["rootAge"]) if args["rootAge"] else None
seqLen = int(args["seqLen"]) if args["seqLen"] else 1000
k = int(args["nbin"]) if args["nbin"] else 100
refTreeFile = args["refTreeFile"]
trueTreeFile = args["trueTreeFile"]
pseudo = float(args["pseudo"]) if args["pseudo"] else 1.0
smpl_times = {}

with open(timeFile,"r") as fin:
    for line in fin:
        name,time = line.split()
        smpl_times[name] = float(time)

if args["clockFile"] is not None:
    omega = []
    phi = []
    with open(args["clockFile"]) as fin:
        for line in fin:
            o,p = line.strip().split()
            omega.append(float(o))
            phi.append(float(p))
    init_rate_distr = multinomial(omega,phi)
else:
    init_rate_distr = None        

with open(intreeFile,"r") as fin:
    with open(outtreeFile,"w") as fout:
        with open(infoFile,"w") as finfo:
            for line in fin:
                tree = read_tree_newick(line)
                nodeIdx = 0
                if args["assignLabel"]:
                    for node in tree.traverse_preorder():
                        if not node.is_leaf():
                            node.set_label("I" + str(nodeIdx))
                            nodeIdx += 1                
                tau,omega,phi = EM_date(tree,smpl_times,root_age=rootAge,s=seqLen,refTreeFile=refTreeFile,k=k,fixed_phi=args["fixedPhi"],fixed_tau=False,pseudo=pseudo,trueTreeFile=trueTreeFile,init_rate_distr=init_rate_distr)
                fout.write(tree.newick() + "\n")       
                for (o,p) in zip(omega,phi):
                    finfo.write(str(o) + " " + str(p) + "\n")
