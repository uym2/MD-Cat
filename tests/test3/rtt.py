from sys import argv
from treeswift import *
from random import choices,sample
from math import log2

def sample_by_depth(tree,nleaf,nsample):
# return a list of newick trees if do_extract is True
# otherwise, return a nested list of labels
    leaf_labels = []
    leaf_weights = []
    for node in tree.traverse_preorder():
        log_p_inv = log2(len(node.children)) if not node.is_leaf() else 0
        if node.is_root():
            node.log_p_inv = log_p_inv
        else:
            node.log_p_inv = node.parent.log_p_inv + log_p_inv
        if node.is_leaf():
            leaf_labels.append(node.label)
            leaf_weights.append(2**(-node.log_p_inv))
    samples = []
    for s in range(nsample):
        sample = set(choices(leaf_labels,weights=leaf_weights,k=nleaf))
        samples.append(list(sample))
    return samples           

def f_rtt(mu,t0,B,T):
    return sum((b-mu*(t-t0))**2 for b,t in zip(B,T))

def optimize_rtt(root_node,smpl_time,pseudo=0):
    T = []
    B = []
    lb2idx = {}
    i = 0
    root_node.d2root = 0
    for node in root_node.traverse_preorder():
        if node is not root_node:
            node.d2root = node.parent.d2root + node.edge_length + pseudo
        if node.label in smpl_time:        
            lb2idx[node.label] = i
            i += 1
            B.append(node.d2root)
            T.append(smpl_time[node.label]) 
    n = len(T)
    s_t = sum(T)
    ss_t = sum(t*t for t in T)
    if s_t*s_t == n*ss_t:
        return None,None,None
    s_b = sum(B)
    s_bt = sum(b*t for b,t in zip(B,T))
    mu = max(0.001,(s_b*s_t-n*s_bt)/(s_t*s_t-n*ss_t))
    t0 = (s_t-s_b/mu)/n
    return mu,t0,f_rtt(mu,t0,B,T)

def bootstrap_rtt(tree,B,T,lb2idx,nsmpl):
    n = len(B)
    #samples = sample_by_depth(tree,n,nsmpl)
    I = range(n)

    #for sample in samples:
        #B_i = [B[lb2idx[x]] for x in sample]
        #T_i = [T[lb2idx[x]] for x in sample]
        #print(optimize_rtt(B_i,T_i))
    for i in range(nsmpl):
        J = choices(I,k=n)    
        B_i = [B[j] for j in J]
        T_i = [T[j] for j in J]
        print(optimize_rtt(B_i,T_i))        

if __name__ == "__main__":
    tree = read_tree_newick(argv[1])
    smpl_time = {}
    with open(argv[2],'r') as fin:
        for line in fin:
            name,t = line.split()
            smpl_time[name] = float(t)
   
    R = []
    for node in tree.traverse_postorder():
        if node.is_leaf():
            node.nleaf = 1
        else:
            node.nleaf = sum(c.nleaf for c in node.children)
        #if node.nleaf >= 3 and node.nleaf <= 5:
        R.append((node,node.nleaf))   
        
    mus = [] 
    ws = []        
    for r,w in R:
        mu,t0,score = optimize_rtt(r,smpl_time,pseudo=0)
        if mu is not None:
            mus.append(mu)            
            ws.append(w)
    #mu,_,_ = optimize_rtt(tree.root,smpl_time)
    print(sum(mus)/len(mus))
    #print(mu)
