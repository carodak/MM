"""Simple classification with scikit_learn."""
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
import minerminor.mm_utils as mmu
import minerminor.mm_representation as mmr
import sklearn as sk
import networkx as nx
import numpy as np
import sklearn.neural_network as nn
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
#import logging
#logging.getLogger('gensim.models.word2vec').setLevel(logging.ERROR) #disable word2vec warnings (to displaying the progress_bar )

#Get the base_path (where bases are saved in JSON format)
parent_parent_path = os.path.dirname(parent_path)
base_path = parent_parent_path+'/Outputs/Bases/JSON/base_planar_k5/learning-planar-minor_18_[0,1]_1000'

#Directory where you save your model
save_path = parent_parent_path+'/Outputs/Models/'

model = nn.MLPClassifier

#
test_size = 0.33

bases_n2v = True #do you want to generate your node2vec bases ? (default = True)
#Load the base in memory

"""Find node2vec hyperparameters"""

learning_base = mmu.load_base(base_path)


p = 1
q = 1


param_grid = { 
    'p': [0.25,0.50,1,2,4],
    'q': [0.25,0.50,1,2,4]
}

if bases_n2v:

	mmr.learning_base_to_node2vec_files(learning_base,1000,p,q)

base_n2v = parent_parent_path+'/Outputs/Bases/node2vec/' 
learning_base = mmu.load_base_n2v(base_n2v) #now our base contains all node2vec matrixes 


"""MLP trains on two arrays: array X of size (n_samples, n_features), 
which holds the training samples represented as floating point feature vectors; and array y of size (n_samples,), 
which holds the target values (class labels) for the training samples"""

# Extract datarows and their associated labels
data_set, label_set = mmu.create_sample_label_classification(learning_base)

# Create training set and tests
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(data_set, label_set, test_size=test_size) #if you want add , test_size=test_size

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf = model()
#print("learning_base : ",learning_base)
#print("x_train : ",x_train)

# Learning model phase
clf.fit(x_train, y_train)

MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

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
print("Model : {0}".format(model))
print("Matrice de confusion : \n{0}".format(mat_conf))
print("{0}".format(report))