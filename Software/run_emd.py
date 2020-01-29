#! /usr/bin/env python

from emd.emd_lib import *
from treeswift import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i","--input",required=True,help="Input trees")
parser.add_argument("-t","--samplingTime",required=True,help="Sampling time")
parser.add_argument("-r","--rootAge",required=False,help="Root age")
parser.add_argument("-o","--output",required=False,help="The output trees with branch lengths in time unit. Default: [input].emDate")
parser.add_argument("-j","--estParam",required=False,help="Write down the estimated parameters (omega and phi) to this file. Default: [input].emParam")
parser.add_argument("-p","--rep",required=False, help="The number of random replicates for initialization. Default: use lsd initialization instead")
parser.add_argument("-l","--seqLen",required=False, help="The length of the sequences. Default: 1000")
parser.add_argument("-f","--refTreeFile",required=False, help="A reference time tree as initial solution. Default: None. LSD will be run internally and used as reference")

args = vars(parser.parse_args())

intreeFile = args["input"]
outtreeFile = args["output"] if args["output"] else (intreeFile + ".emDate")
infoFile = args["estParam"] if args["estParam"] else (intreeFile + ".emParam")

timeFile = args["samplingTime"]
rootAge = float(args["rootAge"]) if args["rootAge"] else None
seqLen = int(args["seqLen"]) if args["seqLen"] else 1000
refTreeFile = args["refTreeFile"]

smpl_times = {}

with open(timeFile,"r") as fin:
    fin.readline()
    for line in fin:
        name,time = line.split()
        smpl_times[name] = float(time)

with open(intreeFile,"r") as fin:
    with open(outtreeFile,"w") as fout:
        with open(infoFile,"w") as finfo:
            for line in fin:
                tree = read_tree_newick(line)
                tau,omega,phi = EM_date(tree,smpl_times,root_age=rootAge,s=seqLen,refTreeFile=refTreeFile)
                fout.write(tree.newick() + "\n")       
                for (o,p) in zip(omega,phi):
                    finfo.write(str(o) + " " + str(p) + "\n")
