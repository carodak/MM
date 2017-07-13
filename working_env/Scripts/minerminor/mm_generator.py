"""Training Base Generator."""

import networkx as nx
from networkx.readwrite import json_graph as js
import random as rdm
from progress_bar import InitBar as ib
import math
import numpy as np
import minerminor.mm_utils as mmu
import minerminor.mm_draw as mmd
import planarity as pl

def choice_first_node(graph):
    """Pick up a node which can form a cycle and a list for the seconde one."""
    node = rdm.choice(graph.nodes())
    possible_nodes_iter = nx.non_neighbors(graph, node)
    possible_nodes = []
    for i in possible_nodes_iter:
        possible_nodes.append(i)

    if possible_nodes:
        return node, possible_nodes
    else:
        return choice_first_node(graph)


def tree_to_1tree(graph):
    # A update avec la fonction nx.non_edge() ...
    """Transform a tree to 1-tree."""
    first_node, possible_nodes = choice_first_node(graph)
    seconde_node = rdm.choice(possible_nodes)
    graph.add_edge(first_node, seconde_node)

    return graph


def certf_tw2(graph):
    """Certif for TW2 production."""
    finish = True
    G = nx.Graph(graph)
    while finish:
        currt_node = None
        currt_node_sec = None
        dic_degree = nx.degree(G)

        if 1 in dic_degree.values():
            currt_node = rdm.choice([i for i in dic_degree if dic_degree[i] == 1])
            currt_node_sec = rdm.choice(list(nx.all_neighbors(G, currt_node)))

        elif 2 in dic_degree.values():
            currt_node = rdm.choice([i for i in dic_degree if dic_degree[i] == 2])
            currt_node_sec = rdm.choice(list(nx.all_neighbors(G, currt_node)))
        else:
            finish = False

        if finish:
            G_ = nx.contracted_edge(G, (currt_node, currt_node_sec), self_loops=False)
            G = nx.Graph(G_)

    return True if len(G) == 1 else False



def learning_base_T21T_generation(rank):
    """Generate T21T learning base with nonisomorphic tree for a given rank."""
    data_set = nx.nonisomorphic_trees(rank, create="graph")
    tree_set, cycle_set = [], []
    pbar = ib()
    total = nx.number_of_nonisomorphic_trees(rank)
    for count, i in enumerate(data_set):
        pbar(count/total * 100)
        tree_set.append(js.node_link_data(i))
        cycle_set.append(js.node_link_data(tree_to_1tree(i)))

    return [tree_set, cycle_set]


def pTree_generation(tree, arr_ptree_rank):
    """From tree make P_tree."""
    # We check the number of available edge to restrict max_cycle.
    edges_set = []
    max_ptree_rank = max(arr_ptree_rank) + 1
    for edge in nx.non_edges(tree):
        edges_set.append(edge)
    if len(edges_set) < max_ptree_rank:
        max_ptree_rank = len(edges_set)
    # We create a set of pTree with a tree in seed.
    ptree_set = []
    current_graph = tree
    for i in range(max_ptree_rank):
        if i in arr_ptree_rank:
            ptree_set.append(nx.Graph(current_graph))
        current_graph = tree_to_1tree(current_graph)

    return ptree_set


def pTree_basic_cycle_generation(nb_nodes, arr_ptree_rank, depth_base):
    """P-Tree Learning base generator with Euler lem."""
    learning_base = []
    for p_rank in arr_ptree_rank:
        ptree_class = []
        # nb_edges = nb_nodes + p_rank - 1
        pbar = ib()
        print("Construction de la classe {0}-Tree : {1} à construire".format(
            p_rank, depth_base))
        tree_set = nx.nonisomorphic_trees(nb_nodes)
        while len(ptree_class) < depth_base:
            try:
                graph = next(tree_set)
            except StopIteration:
                tree_set = nx.nonisomorphic_trees(nb_nodes)
                graph = next(tree_set)

            for i in range(p_rank):
                non_edges = [[x, y] for x, y in nx.non_edges(graph)]
                x, y = rdm.choice(non_edges)
                graph.add_edge(x, y)

            if not mmu.robust_iso(graph, ptree_class):
                pbar(len(ptree_class)/depth_base*100)
                ptree_class.append(graph)

        learning_base.append(ptree_class)

    return learning_base


def learning_base_pTree_generation(nb_nodes, arr_ptree_rank, feature_size):
    """Generate PTree classes with nonisomorphic tree for a given rank."""
    limit_edges = int((math.pow(nb_nodes, 2)-3*nb_nodes+2)/2)
    max_ptree_rank = max(arr_ptree_rank)
    if limit_edges < max_ptree_rank:
        print("Carefully the limitation of basic cycle is {}".format(
                limit_edges))
        max_ptree_rank = limit_edges
    learning_base = [[] for i in arr_ptree_rank]
    tree_itr = nx.nonisomorphic_trees(nb_nodes, create="graph")
    pbar = ib()
    print("""Creation of {0} features in parallel.
          """.format(len(arr_ptree_rank)*feature_size))
    for count, tree in enumerate(tree_itr):
        pbar(count/feature_size*100)
        for i, graph in enumerate(pTree_generation(tree, arr_ptree_rank)):
            learning_base[i].append(graph)
        if count >= feature_size:
            break

    return learning_base


def learning_base_tw2(nb_nodes, arr_tw_rank, feature_size):
    """Generate TW2 learning base."""
    learning_base = [[] for i in arr_tw_rank]
    for count_rank, rank in enumerate(arr_tw_rank):
        print("\nCreation of the TW class : {0}".format(rank))
        pbar = ib()
        for step in range(feature_size):
            pbar((step/feature_size)*100)
            is_good = True
            while is_good:
                G = nx.complete_graph(rank)
                while len(G) < nb_nodes:
                    node_focus = len(G)
                    clique = random_clique(G, rank)
                    G.add_node(node_focus)
                    for i in clique:
                        G.add_edge(node_focus, i)
                if not mmu.robust_iso(G, learning_base[count_rank]):
                    is_good = False
            # print("TW ==> "+str(nx.chordal_graph_treewidth(G)))
            # mmd.show_graph(G)
            learning_base[count_rank].append(G)

    return learning_base


def learning_base_planar_by_minor_agreg(nb_nodes, _, feature_size, minor):
    """Learning base planar by clique agreg."""
    learning_base = [[], []]
    print("\nPlanar class creation")
    pbar = ib()
    for step in range(feature_size):
        pbar((step/feature_size)*100)
        # Creation of P

        # minor creation 
        G = nx.Graph(minor)
        edge = rdm.choice(G.edges())
        G.remove_edge(*edge)
        is_good = True
        while is_good:
            G_ = nx.Graph(G)
            G_.add_node(len(G_))

            edge = rdm.choice(list(G.edges()))
            G_.add_edge(len(G_) - 1, edge[0])
            G_.add_edge(len(G_) - 1, edge[1])

            if pl.is_planar(G_):
                G = nx.Graph(G_)
                if len(G) == nb_nodes:
                    is_good = False
        learning_base[0].append(G)

        G = nx.Graph(minor)
        is_good = True
        while is_good:
            G_ = nx.Graph(G)
            G_.add_node(len(G_))

            edge = rdm.choice(list(G.edges()))
            G_.add_edge(len(G_) - 1, edge[0])
            G_.add_edge(len(G_) - 1, edge[1])

            if not pl.is_planar(G_):
                G = nx.Graph(G_)
                if len(G) == nb_nodes:
                    is_good = False
        learning_base[1].append(G)

    return learning_base


def learning_base_planar(nb_nodes, arr_planar_rank, feature_size):
    """Generate Planar base."""
    learning_base = [[] for i in arr_planar_rank]
    print("\nPlanar class creation..")
    pbar = ib()
    for step in range(feature_size):
        pbar((step/feature_size)*100)
        is_good = True
        while is_good:
            G, H = generate_planar_deg(nb_nodes, step % 2)
            if not mmu.robust_iso(G, learning_base[0]) and not mmu.robust_iso(H, learning_base[1]):
                is_good = False
        learning_base[0].append(G)
        learning_base[1].append(H)

    return learning_base


def learning_base_rdm(nb_nodes, _, feature_size):
    """Generate 0-p/k-p."""
    learning_base = [[], []]
    print("\nTest base creation..; 0-P/k-P")
    learning_base[0] = rdm.sample(list(nx.nonisomorphic_trees(nb_nodes)), feature_size)
    # mmd.show_graph(learning_base[0][0])
    learning_base[1] = agreg_tree(rdm.sample(list(nx.nonisomorphic_trees(nb_nodes)), feature_size), nb_nodes, feature_size)

    return learning_base


def learning_base_rdm_tw2(nb_nodes, _, feature_size):
    """Generate rdm base TW2."""
    learning_base = [[], []]
    tmp = 0
    tree_noniso_list = list(nx.nonisomorphic_trees(nb_nodes))
    print("\nRdm TW2 base creation..")
    pbar = ib()
    while len(learning_base[0]) < feature_size or len(learning_base[1]) < feature_size:
        graph = agreg_tree([rdm.choice(tree_noniso_list)], nb_nodes, feature_size)[0]
        # print(len(tree_noniso_list))
        if certf_tw2(graph):
            if len(learning_base[0]) < feature_size:
                learning_base[0].append(graph)
        else:
            if len(learning_base[1]) < feature_size:
                learning_base[1].append(graph)
        pbar(((len(learning_base[0])+len(learning_base[1]))/(feature_size*2))*100)
        # tmp = 1 if tmp == 0 else 0
        # print(len(learning_base[0]), len(learning_base[1]), tmp)
    return learning_base
        # mmd.show_graph(graph)


def agreg_tree(arr, nb_nodes, feature_size):
    """Agreg tree."""
    min_ = nb_nodes - 1
    max_ = int((nb_nodes * min_) / 2)

    for count, tree in enumerate(arr):
        rdm_edges_int = rdm.randint(0, max_ - len(tree.edges()))
        for i in range(rdm_edges_int):
            # import ipdb; ipdb.set_trace()
            edge = rdm.choice(nx.complement(tree).edges())
            tree.add_edge(*edge)

    return arr


def generate_planar_deg(nb_nodes, side_start):
    """Private method for planar generation."""
    if side_start % 2 == 0:
        G = rdm.choice(list(nx.nonisomorphic_trees(nb_nodes)))
        edges_action = G.add_edge
        set_choice = nx.non_edges
    else:
        G = nx.complete_graph(nb_nodes)
        edges_action = G.remove_edge
        set_choice = nx.edges
    is_good = True
    while(is_good):
        H = G.copy()
        edges_choice = rdm.choice(list(set_choice(G)))
        edges_action(edges_choice[0], edges_choice[1])
        if((side_start % 2 == 0 and not pl.is_planar(G) or (side_start % 2 == 1 and pl.is_planar(G)))):
            is_good = False

    if side_start % 2 == 0:
        return H, G
    else:
        return G, H


def random_clique(G, clique_rank):
    """Return random clique of G."""
    arr_clique = [i for i in nx.enumerate_all_cliques(G) if len(i) == clique_rank]
    return rdm.choice(arr_clique)


def random_path(G, path_size):
    """Return a random path of size n."""
    res = []
    arr_nodes = np.arange(0, len(G))
    focus_node = rdm.choice(arr_nodes)
    res.append(focus_node)
    for i in range(path_size-1):
        arr_neig = [i for i in G.neighbors(focus_node) if i not in res]
        focus_node = rdm.choice(arr_neig)
        res.append(focus_node)

    return res
