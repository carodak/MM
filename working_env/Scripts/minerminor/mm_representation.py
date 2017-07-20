"""Representation function for MinorMiner."""
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
parent_parent_path = os.path.dirname(parent_path)
sys.path.append(parent_path+'/2_Representations/node2vec')
import node2vec
import n2v_main
import numpy as np
import networkx as nx
import math
from sklearn import decomposition
from progress_bar import InitBar 


def learning_base_to_rep(learning_base, arr_rep):
    for count_classe, classe in enumerate(learning_base): #learning_base contains an array of array of graphs [[graph1_P,graph2_P..],[graph1_!P,graph2_!P..]
        for count_graph, graph in enumerate(classe): #for each graph of class P or class !P
            tmp = graph
            for fun in arr_rep:
                tmp = fun(tmp) #use the graph "tmp" as an argument of fun (arr_rep) with this process you can transform the graph into adj, laplacian..
            # rep = rep_2(rep_1(graph))
            learning_base[count_classe][count_graph] = tmp

    return learning_base

""" ---- -----
    First transform our graphs into node2vec matrixes
    save them (in order to do not have to generate them again)
    We'll load them into the learning base in the programm
"""
def learning_base_to_node2vec_files(learning_base, feature_size, p, q, s):
    #first of all, we are going to delete previous node2vec files
    filelist = [ f for f in os.listdir(parent_parent_path+'/Outputs/Bases/node2vec') if f.endswith(".emb")  ]

    for f in filelist:
        os.remove(parent_parent_path+'/Outputs/Bases/node2vec/'+f)
    print("We've deleted all previous node2vec files")

    print("\nTransforming our graphs into node2vec graphs..")

    print("Warn: It can take a long time, do not close this window")

    #pbar = InitBar()
    i = 1
    j = 0
    for count_classes, classes in enumerate(learning_base): #learning_base contains an array of array of graphs [[graph1_P,graph2_P..],[graph1_!P,graph2_!P..]
        for count_graph, graph in enumerate(classes): #for each graph of class P or class !P
            #pbar((count_graph/feature_size)*100)

            tmp = graph
            for edge in tmp.edges():
                tmp[edge[0]][edge[1]]['weight'] = 1
            tmp = tmp.to_undirected()

            G = node2vec.Graph(tmp, False, p, q) #directed=false, p=1, q=1
            G.preprocess_transition_probs()
            walks = G.simulate_walks(10, 80) #number of walks, walks length
            filename = '{0}_{1}'.format(i, j) #we will save our file following this : GraphNumber_Property.emb -> 1_0.emb = graph1 which belongs to property 0
            n2v_main.learn_embeddings(walks, filename, s)
            i = i+1
        print("Finished creating graphs which are in class ", j)
        j = j + 1

    print("Done")
    return None

def vec_to_graph(vec):
    """Vector to graph."""
    root = int(round(math.sqrt(len(vec))))

    return nx.from_numpy_matrix(np.matrix(vec).reshape([root, root]))


def graph_to_A3_minus_D(graph):
    """Convert graph to A3 - D(A2)."""
    mat = nx.to_numpy_matrix(graph)
    mat_2 = mat * mat
    np.fill_diagonal(mat_2, 0)
    res = mat_2 * mat
    diag = np.diag(res)
    idx = np.argsort(diag)[::-1]
    res = res[idx, :][:, idx]

    return np.squeeze(np.asarray(res.reshape(-1)))


def graph_set_to_A3_minus_D(graph_set):
    """Convert a graph set to vector."""
    vec_A3_minus_set = []
    for graph in graph_set:
        vec_A3_minus_set.append(graph_to_A3_minus_D(graph))

    return vec_A3_minus_set


def A3_minus_D(learning_base):
    """Convert blabla."""
    learning_base_A3 = []
    for graph_set in learning_base:
        learning_base_A3.append(graph_set_to_A3_minus_D(graph_set))

    return learning_base_A3
    

def graph_to_vec_adjacency(graph):
    """Convert a graph to a vector from adjacency matrix."""
    mat = nx.to_numpy_matrix(graph)

    return np.squeeze(np.asarray(mat.reshape(-1)))


def graph_to_vec_laplacian(graph):
    """Convert a graph to a vector from laplacian matrix."""
    mat = nx.laplacian_matrix(graph).toarray()

    return np.squeeze(np.asarray(mat.reshape(-1)))


def mat_to_PCA(matrice):
    """Convert to PCA."""
    pca = decomposition.PCA(n_components=len(matrice[0]))
    return pca.fit_transform(matrice)
    # cov = np.cov(matrice.T)
    # ev, eig = np.linalg.eig(cov)
    #
    # return eig.dot(matrice.T)


def mat_to_FP_r(matrice):
    """Convert to FP."""
    q, r = np.linalg.qr(matrice)
    return r


def mat_to_FP_q(matrice):
    """Convert to Q."""
    q, r = np.linalg.qr(matrice)
    return q


def graph_set_to_vec_adjacency_set(graph_set):
    """Convert a graph set to a vector adjacency set."""
    vec_adjacency_set = []
    for graph in graph_set:
        vec_adjacency_set.append(graph_to_vec_adjacency(graph))

    return vec_adjacency_set


def graph_set_to_vec_laplacian_set(graph_set):
    """Convert a graph set to a vector adjacency set."""
    vec_laplacian_set = []
    for graph in graph_set:
        vec_laplacian_set.append(graph_to_vec_laplacian(graph))

    return vec_laplacian_set


def adjacency(labels_set):
    """Convert labels set (set of graph set) to vector adjacency."""
    labels_vec_adjacency = []
    for graph_set in labels_set:
        labels_vec_adjacency.append(graph_set_to_vec_adjacency_set(graph_set))

    return labels_vec_adjacency


def laplacian(labels_set):
    """Convert labels set (set of graph set) to vector laplacian."""
    labels_vec_laplacian = []
    for graph_set in labels_set:
        labels_vec_laplacian.append(graph_set_to_vec_laplacian_set(graph_set))

    return labels_vec_laplacian
