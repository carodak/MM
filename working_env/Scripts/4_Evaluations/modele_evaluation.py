import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import networkx as nx
import keras as ks

parent_parent_path = os.path.dirname(parent_path)
clf_path = parent_parent_path+'/Outputs/Models/model.pkl'
base_path = parent_parent_path+'/Outputs/Bases/JSON/tree_base_test_10_20/learning-base-rdm_10_[0, 1]_20/'
arr_rep = [lambda x: mmr.graph_to_vec_adjacency(x)]
# arr_rep = [lambda x: np.squeeze(np.asarray(nx.laplacian_matrix(x).toarray().reshape(-1)))]

graph_dim = 10
keras_model = False
nbr_classes = 2


# Bases Tasks
learning_base = mmu.load_base(base_path)

learning_base = mmr.learning_base_to_rep(learning_base, arr_rep)

data_set, label_set = mmu.create_sample_label_classification(learning_base)

if keras_model:
    # --reshape for cnn
    data_set, label_set = [np.array(i) for i in [data_set, label_set]]
    data_set = data_set.reshape(data_set.shape[0], graph_dim, graph_dim, 1)
    # convert class vectors to binary class matrices
    label_set_cat = ks.utils.to_categorical(label_set, nbr_classes)

# Classification Tasks
if keras_model:
    clf = ks.models.load_model(clf_path)
else:
    clf = joblib.load(clf_path)

pred_set = clf.predict(data_set)
if keras_model:
    pred_set = [0 if i[0] > i[1] else 1 for i in pred_set]
# classified = (label_set == pred_set).sum()
mat_conf = confusion_matrix(label_set, pred_set)
print(classification_report(label_set, pred_set, target_names=['P', '!P']))
print("\n{0}".format(mat_conf))
