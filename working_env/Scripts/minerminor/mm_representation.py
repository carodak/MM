"""Representation function for MinorMiner."""
import numpy as np
import networkx as nx
import math
from sklearn import decomposition


def learning_base_to_rep(learning_base, arr_rep):
    for count_classe, classe in enumerate(learning_base): #learning_base contains an array of array of graphs [[graph1_P,graph2_P..],[graph1_!P,graph2_!P..]
        for count_graph, graph in enumerate(classe): #for each graph of class P or class !P
            tmp = graph
            for fun in arr_rep:
                tmp = fun(tmp) #use the graph "tmp" as an argument of fun (arr_rep) with this process you can transform the graph into adj, laplacian..
            # rep = rep_2(rep_1(graph))
            learning_base[count_classe][count_graph] = tmp

    return learning_base


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
