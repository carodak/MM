"""Script."""
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
import minerminor.mm_generator as mmg
import minerminor.mm_utils as mmu
import minerminor.mm_draw as mmd
import networkx as nx
import random as rdm

parent_parent_path = os.path.dirname(parent_path)
dir = parent_parent_path+'/Outputs/Bases/JSON/'

if not os.path.exists(dir+"base_planar_k5k33"):
    os.makedirs(dir+"base_planar_k5k33")
    os.makedirs(dir+"base_planar_k5k33/learning-planar-minor_18_[0,1]_50")

if not os.path.exists(dir+"base_planar_k5"):
    os.makedirs(dir+"base_planar_k5")
    os.makedirs(dir+"base_planar_k5/learning-planar-minor_18_[0,1]_50")

if not os.path.exists(dir+"base_planar_k5k33"):
    os.makedirs(dir+"base_planar_k33")
    os.makedirs(dir+"base_planar_k33/learning-planar-minor_18_[0,1]_50")

K5 = nx.complete_graph(5)
K33 = nx.complete_bipartite_graph(3, 3)
lb = mmg.learning_base_planar_by_minor_agreg(18, 50, K33)
lb2 = mmg.learning_base_planar_by_minor_agreg(18, 50, K5)

lb3 = [[], []]

lb3[0].extend(rdm.sample(lb[0], 25))
lb3[0].extend(rdm.sample(lb2[0], 25))

lb3[1].extend(rdm.sample(lb[1], 25))
lb3[1].extend(rdm.sample(lb2[1], 25))

mmu.store_base(lb, dir+"base_planar_k33/learning-planar-minor_18_[0,1]_50")
mmu.store_base(lb2, dir+"base_planar_k5/learning-planar-minor_18_[0,1]_50")
mmu.store_base(lb3, dir+"base_planar_k5k33/learning-planar-minor_18_[0,1]_50")
# mmd.show_graph(lb[0][1])
# mmd.show_graph(lb[1][1])
