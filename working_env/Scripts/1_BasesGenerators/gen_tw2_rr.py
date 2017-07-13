import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
from minerminor import mm_generator as mmg
from minerminor import mm_draw as mmd
from minerminor import mm_utils as mmu
import networkx as nx
import os
# mmg.certf_tw2(a)

feature_size = 1000

learning_base = mmg.learning_base_rdm_tw2(18, None, feature_size)

parent_parent_path = os.path.dirname(parent_path)
dir = parent_parent_path+'/Outputs/Bases/JSON/'

if not os.path.exists(dir+"bases/base_tw2_rdm_test"):
    os.makedirs(dir+"bases/base_tw2_rdm_test")

mmu.store_base(learning_base, dir+"bases/base_tw2_rdm/base_tw2_rdm_test_"+str(feature_size))

# for i in learning_base[0]:
#     mmd.show_graph(i)
