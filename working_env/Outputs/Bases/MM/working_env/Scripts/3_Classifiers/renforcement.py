from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from minerminor import mm_representation as mmr
from minerminor import mm_utils as mmu
import numpy as np
import networkx as nx
import keras as ks

clf_path = "classifier/clf_planar_18/clf_cnn_planar_rdm_adja.h5"
base_renf_path = "bases/base_planar_rdm_test/learning-base-planar_18_[0, 1]_1000"
# arr_rep = [lambda x: np.squeeze(np.asarray(nx.to_numpy_matrix(x).reshape(-1)))]
arr_rep = [lambda x: np.squeeze(np.asarray(nx.laplacian_matrix(x).toarray().reshape(-1)))]

keras_model = True
graph_dim = 18
nbr_classes = 2

learning_base = mmu.load_base(base_renf_path)
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


y_pred = cross_val_predict(clf, data_set, label_set, cv=10)

mat_conf = confusion_matrix(label_set, y_pred)
report = classification_report(label_set, y_pred, target_names=['P', '!P'])

print("{0}|{1}|\n\n{2}|\n{3}|\n{4}".format(base_renf_path,
                                           arr_rep,
                                           clf,
                                           mat_conf.tolist(),
                                           report))
