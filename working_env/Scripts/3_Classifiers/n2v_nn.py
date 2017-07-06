"""Simple classification with scikit_learn."""
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
from sklearn.externals import joblib

#Get the base_path (where bases are saved in JSON format)
parent_parent_path = os.path.dirname(parent_path)
base_path = parent_parent_path+'/Outputs/Bases/JSON/tree_base_test_10_20/learning-base-rdm_10_[0, 1]_20'


test_size = 0.2
model = sk_tree.DecisionTreeClassifier
#Directory where you save your model
save_path = parent_parent_path+'/Outputs/Models/'

#Load the base in memory
learning_base = mmu.load_base(base_path)

i = 1
for count_classes, classes in enumerate(learning_base): #learning_base contains an array of array of graphs [[graph1_P,graph2_P..],[graph1_!P,graph2_!P..]
	for count_graph, graph in enumerate(classes): #for each graph of class P or class !P
		tmp = graph
		for edge in tmp.edges():
			tmp[edge[0]][edge[1]]['weight'] = 1
		tmp = tmp.to_undirected()

		G = node2vec.Graph(tmp, False, 1, 1) #directed=false, p=1, q=1
		G.preprocess_transition_probs()
		walks = G.simulate_walks(10, 80) #number of walks, walks length
		filename = 'n2v_G%d' % i
		n2v_main.learn_embeddings(walks, filename)
		i = i+1


#A partir de la base networkx, on créé des fichiers qui contiennent nos graphes donnés par liste d'arêtes afin d'utiliser la représentation node2vec qui prend en entrée une edgelist (déplacer ce code dans une fonction de utils)
"""i = 1
for count_classe, classe in enumerate(learning_base): #learning_base contains an array of array of graphs [[graph1_P,graph2_P..],[graph1_!P,graph2_!P..]
        for count_graph, graph in enumerate(classe): #for each graph of class P or class !P
            tmp = graph
            #on créé un fichier de liste d'arêtes pour chaque graph
            write_edgelist = nx.write_edgelist(tmp,"/home/carodak/Documents/Stage/minerminor-master/virtualenvs/my_project/working_env/bases/base_arbre/edgelist/edgelist_base(%d).edgelist" % i)
            i = i+1 """


