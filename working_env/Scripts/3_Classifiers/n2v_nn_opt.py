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
base_path = parent_parent_path+'/Outputs/Bases/JSON/base_planar_k5/learning-planar-minor_18_[0,1]_50'

#Directory where you save your model
save_path = parent_parent_path+'/Outputs/Models/'

model = nn.MLPClassifier

#
test_size = 0.2



"""Find node2vec hyperparameters
p and q : node2vec parameters to choose between Breadth First Search and Depth First Search
s = matrix size (width)

You can specify the range of your p and q values but also of your matrix size 
walks_n = number of walks
walks_l = lengh of each walk
"""

hyperparameters = []

x = np.arange(1, 20, 1)
y = np.arange(3, 6, 1)
walks_n = 10
walks_l = 80

for p in x:
  for q in x:
    for s in y:
      learning_base = mmu.load_base(base_path) #load the bases of JSON files, load the base in memory
      mmr.learning_base_to_node2vec_files(learning_base,50,p,q,s)

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
      report2 = sk.metrics.precision_recall_fscore_support(y_test, y_pred, average='weighted')

      print("Base_path : {0}".format(base_path))
      print("Model : {0}".format(model))
      print("p : {0}, q : {1}, s : {2}".format(p,q,s))
      print("Confusion matrix : \n{0}".format(mat_conf))
      print("{0}".format(report))

      f_measure = report2[2]
        
      mat = []
      mat.append(p)
      mat.append(q)
      mat.append(s)
      mat.append(f_measure)

      hyperparameters.append(mat)

      print("hyperparameters",hyperparameters)


    """
    Once we have finished testing hyperparameters
    Let's print what we found and save it into a file
    """

    print("\n .=======(((======= Victory ======))======.")
    print("We've finished testing hyperparameters.")
    print("Let's see what we found.")

    best_measure = 0.0
    p = 0
    q = 0
    s = 0


    for h in hyperparameters:
      if h[3]>best_measure:
        best_measure = h[3]
        p = h[0]
        q = h[1]
        s = h[2]

    print("Best parameters are : p: {0}, q: {1} s: {2} with a precision of {3} ".format(p,q,s,best_measure))
    np.savetxt(parent_parent_path+'/Outputs/Results/node2vec/best_hyperparameters.txt', hyperparameters)


