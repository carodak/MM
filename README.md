# minerminor
Minerminor library

> let's have some fun !!!

OBJECTIVE : We wanna predict if a graph has a given property

First of all, you need to install following libraries. We highly recommand you to install them in virtual environnement

- Scikit-learn
- TensorFlow
- Keras
- Networkx

STEPS: 

1) GRAPHS GENERATION

Machine learning is all about data so we will generate a lot of graphs, and stock them into 2 bases (classes): 
	- graphs that have the property P (they will be saved as label "1" vice versa): for example Planar Graphs 
	- graphs that have not the property P (they will be saved as label "0" vice versa): for example Non-Planar Graphs

You can find scripts to generate these bases by following this path : working_env/Scripts/1_BasesGenerators
but also in the minerminor library : working_env/Scripts/minerminor/mm_generator.py

HOW TO RUN IT?

example: (virtualenv) user@user:~/../MM$ python3 working_env/Scripts/1_BasesGenerators/gen_ptree_bench.py -n 10 -s 20 -p tree_base_test_10_20

Outputs: 

Outputs graphs will be saved as JSON files. You can find them by following this path : working_env/Outputs/Bases/JSON/tree_base_test_10_20/learning-base-rdm_10_[0, 1]_20/

Library: 

Before be saved as JSON files our graphs were networkx graphs

2) CHOOSE THE REPRESENTATION BUT ALSO THE CLASSIFIER

We have combined these steps into one script. 

Representation: we're transforming one networkx graph into something easier to manipulate (for classifiers) like laplacian matrix, adjacency matrix, node2vec graphs...
Attention: Choosing the appropriate representation according to the classifier is an important step!

Classifier: 

After chosen the appropriate representation, you have to choose a model to classify your data (classifier). 
For example you can pass your matrixes to a NN or CNN model

If you choose CNN, your matrixes would be your training data. At the end of the classification the script would print the results of the prediction task.

TODO
Inside of the file (working_env/Scripts/3_Classifiers/simple_classification.py): 

specify the generated base: base_path = parent_parent_path+'/Outputs/Bases/JSON/tree_base_test_10_20/learning-base-rdm_10_[0, 1]_20'
specify the data representation: rep_adja = lambda x: mmr.graph_to_vec_adjacency(x)
save the model to evaluate it later: joblib.dump(clf, parent_parent_path+'/Outputs/Models/model.pkl') 
 
Once these steps done, you can run the chosen classifier:

(virtualenv) user@user:~/../MM$ python3 working_env/Scripts/3_Classifiers/simple_classification.py 

Warning : if you choose n2v representation, you may have word2vec library error with python 3
you need to modify your word2vec.py -> return sum(len(list(sentence)) for sentence in job)

3) EVALUATE YOUR MODEL:

Now, you've got your trained model and you want to know if your model can predict "if any given graph has a property P".

Inside of the file (working_env/Scripts/4_Evaluations/modele_evaluation.py):
specify saved model : clf_path = parent_parent_path+'/Outputs/Models/model.pkl'
base : base_path = parent_parent_path+'/Outputs/Bases/JSON/tree_base_test_10_20/learning-base-rdm_10_[0, 1]_20/'
representation : arr_rep = [lambda x: mmr.graph_to_vec_adjacency(x)]
number of nodes in graph = 10 nodes
number of classes : 2 (class 0 or class 1)

Once they steps done, run the script :
(virtualenv) user@user:~/../MM$ python3 working_env/Scripts/4_Evaluations/modele_evaluation.py
