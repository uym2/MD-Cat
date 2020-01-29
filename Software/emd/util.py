from bitsets import bitset
from treeswift import *

def bitset_from_tree(tree):
    taxa = tuple(node.label for node in tree.traverse_leaves())
    BS = bitset('BS',taxa)
    return BS

def bitset_index(tree,BS):
# the tree MUST have the same taxa as those that are encoded in BS
    for node in tree.traverse_postorder():
        if node.is_leaf():
            node.bits = BS([node.label])
        else:
            C = node.child_nodes()
            bs =  C[0].bits
            for c in C[1:]:
                bs = bs.union(c.bits)
            node.bits = bs     
    
def bitset_index_trees(trees):
    BS = bitset_from_tree(trees[0])

    for tree in trees:
        bitset_index(tree)        
