"""Testing script."""
# from minerminor import mm_generator as mmg
from minerminor import mm_utils as mmu
from minerminor import mm_draw as mmd
from minerminor import mm_representation as mmr
from sklearn import ensemble, svm, neighbors, tree
from sklearn.neural_network import MLPClassifier

# import networkx as nx
# import numpy as np
from sklearn.model_selection import train_test_split
# graph = nx.from_numpy_matrix(np.matrix([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]]))
# vec = mmr.graph_to_vec_adjacency(graph)
# print(vec)
# mmu.show_graph(mmr.vec_to_graph(vec))

learning_base = mmu.load_base("base_rdm_kp/learning-base-rdm_14_[0, 1]_1000")
# learning_base = mmr.A3_minus_D(learning_base_brute)
data_set, label_set = mmu.create_sample_label_classification(learning_base)

X_train, X_test, y_train, y_test = train_test_split(data_set, label_set,
                                                    test_size=0.1)
X_train_rep = mmr.graph_set_to_vec_laplacian_set(X_train)
X_test_rep = mmr.graph_set_to_vec_laplacian_set(X_test)

clf = MLPClassifier()

# print(len(X_test[1]))
pred, miss, clf = mmu.learning(clf, X_train_rep, X_test_rep, y_train, y_test)
to_show = mmu.extract_miss_class(y_test, pred)

for x in to_show:
    mmd.show_graph(X_test[x])
