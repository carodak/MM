from minerminor import mm_generator as mmg
from minerminor import mm_draw as mmd
from minerminor import mm_utils as mmu
import networkx as nx
import os
# mmg.certf_tw2(a)

feature_size = 1000

learning_base = mmg.learning_base_rdm_tw2(18, None, feature_size)

if not os.path.exists("bases/base_tw2_rdm_test"):
    os.makedirs("bases/base_tw2_rdm_test")

mmu.store_base(learning_base, "bases/base_tw2_rdm/base_tw2_rdm_test_"+str(feature_size))

# for i in learning_base[0]:
#     mmd.show_graph(i)
