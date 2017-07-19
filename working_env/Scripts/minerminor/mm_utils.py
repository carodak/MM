"""Tools Box MinorMiner."""
import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from networkx.readwrite import json_graph
import time
import csv
import datetime
from sklearn.model_selection import learning_curve
import random as rdm
import csv
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import re
# from minerminor import mm_representation as mmr


def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

#explore the iterable, add 1 at each iteration -> return the len of the iterable
def count_iterable(i): 
    """len() pour un iterable."""
    return sum(1 for e in i)


def info_graph(graph):
    """Describe the argument graph."""
    print(nx.info(graph)) #info(G[, n]) : Print short summary of information for the graph G or the node n.
    print(nx.cycle_basis(graph)) 
    """cycle_basis: Returns a list of cycles which form a basis for cycles of G.
    A basis for cycles of a network is a minimal collection of cycles such that any cycle in the network 
    can be written as a sum of cycles in the basis.
    Example :
    G=nx.Graph()
    G.add_cycle([0,1,2,3])
    G.add_cycle([0,3,4,5])
    print(nx.cycle_basis(G,0))
    [[3, 4, 5, 0], [1, 2, 3, 0]]
    """
    print(nx.laplacian_spectrum(graph)) #laplacian_spectrum(G) : Return eigenvalues of the Laplacian of G

#way to know if two graphs aren't isomorphic
def fast_iso(graph, graph_set):
    """Fast iso test."""
    for g_test in graph_set:
        if nx.could_be_isomorphic(graph, g_test): # return false if these graphs are not isomorphic, true doesn't garantee the isomorphism
            return True
    return False

#Returns True if graph and graph_set are isomorphic graphs.
def robust_iso(graph, graph_set):
    """Fast iso test."""
    for g_test in graph_set:
        if nx.is_isomorphic(graph, g_test):
            return True
    return False


def graph_sampling(graph, rank_sub):
    """Graph sub sampling."""
    res = [rdm.choice(graph.nodes())] #random.choice: Return a random element from the non-empty sequence "graph.nodes()"
    """range : This is a versatile function to create lists containing arithmetic progressions. 
    range(10)->[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"""
    for i in range(rank_sub - 1): 
        seed = []
        for j in res: 
            seed += [k for k in nx.all_neighbors(graph, j)] #seed will contain all neighbors of the random node
        seed = [l for l in list(set(seed)) if l not in res] #for each neighbor, check if it's equal to res
        res.append(rdm.choice(seed)) #if not : add a random element which belongs to seed (all neighbors of the random node) to res

    return nx.subgraph(graph, res)
    """ Basically, we'll choose a node randomly then we'll enumerate all of its neighbors 
    then choose one of its neighbors and enumerate all neighbors of the chosen node...""" 


def graph_set_to_json(graph_set):
    """Convert graph set to json set."""
    json_set = []
    for g in graph_set:
        json_set.append(json_graph.node_link_data(g))

    return json_set


def json_to_graph_set(json_set):
    """Convert json set to graph set."""
    graph_set = []
    for j in json_set:
        graph_set.append(json_graph.node_link_graph(j))

    return graph_set


def store_base(labels_set, label_set_name):
    """Store the learning base on JSON files."""
    if not os.path.exists(label_set_name):
        os.makedirs(label_set_name)
    for i, g in enumerate(labels_set):
        print("Writting {0} graphs with the label {1}.".format(len(g), i))
        file_path = "{0}/{1}_label_base.json".format(label_set_name, i)
        with open(file_path, "w") as f:
            json.dump(graph_set_to_json(g), f)


def load_base(label_set_name):
    """Load the learning base from JSON files."""
    list_labels_files = os.listdir(label_set_name) #listdir return a list of all the files of the folder (argument)
    labels_set = []
    for file_name in list_labels_files: #for every files in the folder
        path = "{0}/{1}".format(label_set_name, file_name) #path of one file
        with open(path) as f: #open the file which has "path" as path
            list_gaph = json.load(f) #save json list of graphs into "list_graph"
            label_set = []
            for graph in list_gaph: #for each graph
                label_set.append(json_graph.node_link_graph(graph))
            labels_set.append(label_set)
    #print("labals_set: ",labels_set)
    return labels_set



def load_base_n2v(label_set_name):
    """Load the learning base from CSV N2V."""
    learning_base = [[], []] #[ [ [[N1L1,N2L1][N1L2 N2L2]],[[]..], [ [[] []], [[] []] ] ]

    #for each n2v file
    list_labels_files = [f for f in sorted_nicely(os.listdir(label_set_name)) if os.path.isfile(os.path.join(label_set_name, f))]
    for files_name in list_labels_files:
        arr_file = re.split("[_.]+", files_name) #21_1.emb -> arr_file['21', '1', 'emb']
        #print("arr_file: ",arr_file)
        with open(os.path.join(label_set_name, files_name)) as f: #obtain the full path of the selected file then open it
            incsv = csv.reader(f, delimiter=' ') #read the file and obtain numbers of the matrix via the delimiter ' '
            next(incsv)
            mat = []
            for row in incsv: #for each row of the file
                mat.extend(row[1:])
                #mat.append(row[1:])
               
            learning_base[int(arr_file[1])].append(np.array(mat,dtype='float')) #we'll devide our node2vec learning base into 2 classes like it was with JSON files

    #print("learning_base : ",learning_base)

    return learning_base


def tw_split_base(base_path):
    """Transform unebase {C1, C2, ... Cn} en {(C1uC2uCn-1),Cn}."""
    old_learning_base = load_base(base_path)

    if len(old_learning_base) < 2:
        print("Base trop petit, au moins 2 classes")
        return 0

    learning_base = [[], old_learning_base[-1]]
    nb_sample = int(len(old_learning_base[0])/len(old_learning_base[:-1]))
    for feature in old_learning_base[:-1]:
        learning_base[0] = list(set(learning_base[0]) | set(rdm.sample(feature,nb_sample)))

    for i in range(len(learning_base[1])-len(learning_base[0])):
        rdm_len_feature = rdm.randint(0,len(old_learning_base[:-1])-1)
        rdm_len_feature_size = rdm.randint(0,len(old_learning_base[0])-1)
        print(rdm_len_feature, rdm_len_feature_size)
        learning_base[0].append(old_learning_base[rdm_len_feature][rdm_len_feature_size])

    return learning_base



def exp_to_csv(generator, nb_nodes, feature_size, ptree_rank, time_exe):
    """Write the csv of experiment."""
    c = csv.writer(open("{0}.csv".format(datetime.datetime.now)), "wb")
    c.writerow([generator, nb_nodes, feature_size, ptree_rank, time_exe])


def create_sample_label_classification(labels_set):
    """Create data and label set."""
    data_set, label_set = [], []
    for i, graph_set in enumerate(labels_set):
        for graph in graph_set:
            data_set.append(graph)
            label_set.append(i)

    return data_set, label_set


def extract_miss_class(y_test, y_pred):
    """Extract ind element miss classified from exp."""
    miss_graph = []
    for count, value in enumerate(y_test):
        if value != y_pred[count]:
            miss_graph.append(count)

    return miss_graph


def learning(classifier, X_train, X_test, y_train, y_test):
    """Learning and Predict function."""
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    success = (y_test == y_pred).sum()
    total = len(X_test)
    print("Number of labeled on a total %d : %d (%d %%)" % (
        total, success, (success/total)*100))

    return y_pred, (success/total)*100, classifier


def experiment_generation(arr_generator, arr_nb_nodes, arr_ptree_rank,
                          arr_features_size, path_dir):
    """Experiment methode for base learning generation."""
    with open(path_dir+"_generation_results.txt", 'w') as f:
        for generator in arr_generator:
            for nb_nodes in arr_nb_nodes:
                for feature_size in arr_features_size:
                    t0 = time.time()
                    learning_base = generator(nb_nodes,
                                              arr_ptree_rank,
                                              feature_size)
                    t1 = time.time()
                    gen_norm = generator.__name__.replace("_", "-")
                    path = "{0}/{1}_{2}_{3}_{4}/".format(path_dir,
                                                         gen_norm,
                                                         nb_nodes,
                                                         arr_ptree_rank,
                                                         feature_size)
                    store_base(learning_base, path)
                    t2 = time.time()
                    print("""
~~~~~~~~~
Gen : {0},\nNb_nd : {1},\nP-T_rk : {2},\nF_size : {3}, #F : {4}\n
    => Time build : {5} sec, Time store : {6} sec
#########        """.format(generator.__name__,
                            nb_nodes,
                            arr_ptree_rank,
                            feature_size,
                            len(learning_base),
                            str(t1-t0),
                            str(t2-t1)))
                    f.write("{0}|{1}|{2}|{3}|{4}|{5}\n".format(generator.__name__,
                                                               nb_nodes,
                                                               feature_size,
                                                               arr_ptree_rank,
                                                               str(t1-t0),
                                                               str(t2-t1)))
                    # for i, ptree in enumerate(arr_ptree_rank):
                    #     show_graph(learning_base[i][0])


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """learningCurve."""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
