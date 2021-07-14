#! /usr/bin/env python

from treeswift import *
from sys import argv

infile = argv[1]
outfile = argv[2]

lb2sum = {}
n = 0

with open(infile,'r') as fin:
    for line in fin:
        tree = read_tree_newick(line)
        for node in tree.traverse_preorder():
            if not node.is_root():
                lb = node.get_label()
                lb2sum[lb] = node.get_edge_length() if lb not in lb2sum else lb2sum[lb] + node.get_edge_length()
        n += 1
                
for node in tree.traverse_preorder():
    if not node.is_root():
        node.set_edge_length(lb2sum[node.get_label()]/n)
    
tree.write_tree_newick(outfile)                                    
