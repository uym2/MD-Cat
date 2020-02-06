from treeswift import *
from simulator.multinomial import *
from scipy.stats import poisson

eps=0.01

def simulate_poisson(timeTree,rate_distr,seqLen):
# the timeTree must have edges in unit of time
# rate_distr is an instance of multinomial class
# each node in the timeTree will have an extra attribute b which is the branch length in substitution unit
    simulated_mu = []
    for node in timeTree.traverse_preorder():
        if node.is_root():
            continue
        # randomly draw a mutation rate from the distribution
        mu = rate_distr.randomize()
        simulated_mu.append(mu)
        # simulate the Poisson process
        tau = node.edge_length
        ld = seqLen*mu*tau # this is the parameter of the Poisson distribution
        node.b = (eps/seqLen + poisson.rvs(ld))/seqLen

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


        
        

       
