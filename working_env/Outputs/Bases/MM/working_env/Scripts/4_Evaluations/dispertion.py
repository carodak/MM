"""Même nombre de classes et de sommets!!!."""

import networkx as nx
from minerminor import mm_utils as mmu
from minerminor import mm_draw as mmd
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(prog="Graph dispertion")
parser.add_argument("-b", "--base_path", help="path of bases")
args = parser.parse_args()


data_dict = {}
for directory in os.listdir(args.base_path):
    learning_base = mmu.load_base("{0}/{1}".format(args.base_path, directory))
    arr_base = directory.split('_')

    method = arr_base[0]
    nb_nodes = int(arr_base[1])
    feature_size = int(arr_base[3])
    arr_classes = arr_base[2][1:-1].split(',')

    if int(arr_classes[0]) not in data_dict.keys():
        for classe in arr_classes:
            data_dict[int(classe)] = {}

    if 0 not in data_dict[int(arr_classes[0])].keys():
        for classe in arr_classes:
            for i in range(nb_nodes):
                data_dict[int(classe)][i] = {}

    import pdb; pdb.set_trace()
    for classe in arr_classes:
        for i in range(nb_nodes):
            if method not in data_dict[int(classe)][i].keys():
                data_dict[int(classe)][i][method] = {}

    for count_classe, classe in enumerate(learning_base):
        glob = {}
        for elemt in classe:
            eigen_vec = nx.eigenvector_centrality_numpy(elemt)
            for count_eigen, eigen_value in enumerate(eigen_vec.values()):
                # import pdb; pdb.set_trace()
                if count_eigen in glob.keys():
                    glob[count_eigen].append(eigen_value)
                else:
                    glob[count_eigen] = [eigen_value]
        class_eigen_mean_vec = [np.mean(i) for i in glob.values()]
        for count_y, y in enumerate(class_eigen_mean_vec):
            data_dict[count_classe][count_y][method][feature_size] = y

# print (data_dict)
# print(data_dict)
mmd.create_spread_curve(data_dict)
