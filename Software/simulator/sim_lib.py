from treeswift import *
from simulator.multinomial import *
from scipy.stats import poisson, norm
from math import sqrt

eps=0.01

def simulate_scale(timeTree,rate_distr,seqLen=None):
# the timeTree must have edges in unit of time
# rate_distr is an instance of multinomial class
# each node in the timeTree will have an extra attribute b which is the branch length in substitution unit
    simulated_mu = {}
    for node in timeTree.traverse_preorder():
        if node.is_root():
            continue
        # randomly draw a mutation rate from the distribution
        mu = rate_distr.randomize()
        simulated_mu[node.get_label()] = mu
        # simulate the Poisson process
        tau = node.edge_length
        node.b = mu*tau

    return simulated_mu

def simulate_gaussian(timeTree,rate_distr=None,seqLen=1000):
# Goal: simulate the estimated branches in subsitution unit
# if rate_distr is None, set the mutation rate to 1.0 on all branches
# otherwise, rate_distr must be an instance of multinomial, exponential, or lognormal class
# branch length uncertanties are modeled using Gaussian model.
# Output: each node in the timeTree will have an extra attribute b which is the branch length in substitution unit
    simulated_mu = {}
    for node in timeTree.traverse_preorder():
        if node.is_root():
            continue
        # randomly draw a mutation rate from the distribution
        mu = rate_distr.randomize() if rate_distr else 1.0
        simulated_mu[node.get_label()] = mu
        # simulate the Gaussian error
        tau = node.edge_length
        ld = mu*tau # this is the mean of the distribution
        std = sqrt(mu*tau/seqLen)
        while 1:             
            r = norm.rvs(ld,std)
            if r > 1e-8:
                break
        node.b = r        
    return simulated_mu

def simulate_poisson(timeTree,rate_distr=None,seqLen=1000):
# Goal: simulate the estimated branches in subsitution unit
# if rate_distr is None, set the mutation rate to 1.0 on all branches
# otherwise, rate_distr must be an instance of multinomial, exponential, or lognormal class
# branch length uncertanties are modeled using Poisson model.
# Output: each node in the timeTree will have an extra attribute b which is the branch length in substitution unit
    simulated_mu = {}
    for node in timeTree.traverse_preorder():
        if node.is_root():
            continue
        # randomly draw a mutation rate from the distribution
        mu = rate_distr.randomize() if rate_distr else 1.0
        simulated_mu[node.get_label()] = mu
        # simulate the Poisson process
        tau = node.edge_length
        ld = seqLen*mu*tau # this is the parameter of the Poisson distribution
        #node.b = (eps/seqLen + poisson.rvs(ld))/seqLen
        node.b = poisson.rvs(ld)/seqLen
    return simulated_mu

def write_tree(tree,edge_type='b',outfile=None,append=False):
    if outfile:
        outstream = open(outfile,'a' if append else 'w')
    else:
        outstream = stdout

    __write__(tree.root, outstream,edge_type=edge_type)
    outstream.write(";\n")
    if outfile:
        outstream.close()

def __write__(node,outstream,edge_type='b'):
    if node.is_leaf():
        outstream.write(node.label)
    else:
        outstream.write('(')
        is_first_child = True
        for child in node.children:
            if is_first_child:
                is_first_child = False
            else:
                outstream.write(',')
            __write__(child,outstream)
        outstream.write(')')
        if node.label is not None:
            outstream.write(str(node.label))
    
    if not node.parent is None:
        el = node.b if edge_type=='b' else node.edge_length
        outstream.write(":" + str(el))


        
        

       
