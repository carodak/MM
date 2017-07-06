"""Simple classification with scikit_learn."""
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path+'/minerminor')
import minerminor.mm_utils as mmu
import minerminor.mm_representation as mmr
import sklearn as sk
import sklearn.tree as sk_tree
import networkx as nx
from sklearn.externals import joblib

# Get the base_path (where bases are saved in JSON format)
parent_parent_path = os.path.dirname(parent_path)
base_path = parent_parent_path+'/Outputs/Bases/JSON/tree_base_test_10_20/learning-base-rdm_10_[0, 1]_20'
# Choose your representation
rep_adja = lambda x: mmr.graph_to_vec_adjacency(x)
representation_array = [rep_adja]
test_size = 0.2
model = sk_tree.DecisionTreeClassifier
save_path = parent_parent_path+'/Outputs/Models/'
# Load base in memory.
learning_base = mmu.load_base(base_path)
# Apply the representation
learning_base = mmr.learning_base_to_rep(learning_base, representation_array)
# Extract datarows and their associated labels
data_set, label_set = mmu.create_sample_label_classification(learning_base)
# Create training set and tests
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(data_set, label_set, test_size=test_size)
# Instanciation of classification model
clf = model()
# Learning model phase
clf.fit(x_train, y_train)
# Save the model if destination path is defined
if save_path:
    joblib.dump(clf, save_path+'model.pkl')

# Testing the model on the training set
y_pred = clf.predict(x_test)
# Generate confusion matrix
mat_conf = sk.metrics.confusion_matrix(y_test, y_pred)
# Generate test report
report = sk.metrics.classification_report(y_test, y_pred, target_names=['P', '!P'])

print("Base_path : {0}".format(base_path))
print("Representation : {0}".format(representation_array))
print("Model : {0}".format(model))
print("Matrice de confusion : \n{0}".format(mat_conf))
print("{0}".format(report))