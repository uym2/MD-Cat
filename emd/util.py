from bitsets import bitset
from treeswift import *
import re
from datetime import datetime,timedelta

# convert date (YYYY-MM-DD or YYYY-MM) to years since Jan 1, 2019
def date_to_years(sample_time):
    return date_to_days(sample_time)/365

# convert years since Jan 1, 2019 to date (YYYY-MM-DD)
def years_to_date(years):
    days = years*365
    return (datetime(2019,1,1) + timedelta(days=days)).strftime('%Y-%m-%d')

# convert date (YYYY-MM-DD or YYYY-MM) to days since Jan 1, 2019
def date_to_days(sample_time):
    try:
        sample_time = datetime.strptime(sample_time, '%Y-%m-%d')
    except:
        sample_time = datetime.strptime(sample_time, '%Y-%m')
    return (sample_time - datetime(2019,1,1)).days # days since Jan 1, 2019

# convert days since Jan 1, 2019 to date (YYYY-MM-DD)
def days_to_date(days):
    return (datetime(2019,1,1) + timedelta(days=days)).strftime('%Y-%m-%d')

def bitset_from_tree(tree):
    taxa = tuple(node.label for node in tree.traverse_leaves())
    BS = bitset('BS',taxa)
    return BS

def bitset_index(tree,BS):
# the tree MUST have the same taxa as those that are encoded in BS
    for node in tree.traverse_postorder():
        if node.is_leaf():
            lb = re.sub("\[.*\]","",node.label)
            node.bits = BS([lb])
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
