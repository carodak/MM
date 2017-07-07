import networkx as nx
import minerminor.mm_utils as mmu
import minerminor.mm_draw as mmd
import minerminor.mm_representation as mmr
import minerminor.mm_generator as mmg
import numpy as np
from sklearn.externals import joblib


# G = nx.gnm_random_graph(20, 29)
# mmd.show_graph(G)
#
# for i in range(5):
#     G_ = mmu.graph_sampling(G, 10)
#
#     mmd.show_graph(G_)

# input
learning_base_path = "base_sampling/pTree-basic-cycle-generation_20_[0, 1]_10"
clf_path = "classifier/SVC_laplacian_2000.pkl"
representation = mmr.graph_to_vec_laplacian
G_node_size = 20
Gp_node_size = 14
sampling_step = 10

# instance
learning_base = mmu.load_base(learning_base_path)
clf = joblib.load(clf_path)
results_base = [[] for i in learning_base]

# sampling
for count_classe, classe in enumerate(learning_base):
    for instance in classe:
        res = 0
        for step in range(sampling_step):
            G_ = mmu.graph_sampling(instance, Gp_node_size)
            # representation
            G_mat = representation(G_)
            # classification
            if clf.predict(G_mat):
                res = 1
                break
        results_base[count_classe].append(res)

print(results_base)
