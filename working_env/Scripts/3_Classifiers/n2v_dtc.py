"""Decision tree classifier using node2vec"""
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
sys.path.append(parent_path+'/2_Representations/node2vec')
import minerminor.mm_utils as mmu
import minerminor.mm_representation as mmr
import sklearn as sk
import sklearn.tree as sk_tree
import networkx as nx
import node2vec
import n2v_main
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

#Get the base_path (where bases are saved in JSON format)
parent_parent_path = os.path.dirname(parent_path)
base_path = parent_parent_path+'/Outputs/Bases/JSON/tree_base_test_18_20/learning-base-rdm_18_[0, 1]_20'

#Directory where you save your model
save_path = parent_parent_path+'/Outputs/Models/'

model = sk_tree.DecisionTreeClassifier

#
test_size = 0.2
#Load the base in memory

learning_base = mmu.load_base(base_path)

i = 1
j = 0
for count_classes, classes in enumerate(learning_base): #learning_base contains an array of array of graphs [[graph1_P,graph2_P..],[graph1_!P,graph2_!P..]
	for count_graph, graph in enumerate(classes): #for each graph of class P or class !P

		tmp = graph
		for edge in tmp.edges():
			tmp[edge[0]][edge[1]]['weight'] = 1
		tmp = tmp.to_undirected()

		G = node2vec.Graph(tmp, False, 1, 1) #directed=false, p=1, q=1
		G.preprocess_transition_probs()
		walks = G.simulate_walks(10, 80) #number of walks, walks length
		filename = '{0}_{1}'.format(i, j) #we will save our file following this : GraphNumber_Property.emb -> 1_0.emb = graph1 which belongs to property 0
		n2v_main.learn_embeddings(walks, filename)
		i = i+1

	j = j + 1

	#output_path = parent_parent_path+'/Outputs/Bases/node2vec/' % j

	#if not os.path.exists(output_path): # we would like to continue separating our 2 properties
    #		os.makedirs(output_path)

base_n2v = parent_parent_path+'/Outputs/Bases/node2vec/' 

learning_base = mmu.load_base_n2v(base_n2v) #now our base contains all node2vec matrixes 


"""MLP trains on two arrays: array X of size (n_samples, n_features), 
which holds the training samples represented as floating point feature vectors; and array y of size (n_samples,), 
which holds the target values (class labels) for the training samples"""

# Extract datarows and their associated labels
data_set, label_set = mmu.create_sample_label_classification(learning_base)

# Create training set and tests
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(data_set, label_set, test_size=test_size)

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = model()

#print("learning_base : ",learning_base)
#print("x_train : ",x_train)


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
print("Model : {0}".format(model))
print("Matrice de confusion : \n{0}".format(mat_conf))
print("{0}".format(report))

