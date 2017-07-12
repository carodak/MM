"""Test tree classification."""
# import networkx as nx
# import numpy as np
# from minerminor import mm_generator as mmg
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from sklearn import ensemble, svm, neighbors, tree
from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.model_selection import ShuffleSplit


parser = argparse.ArgumentParser(prog="P-Tree by DescTree classification")
parser.add_argument("-r", "--representation", nargs='*',
                    default=[mmr.adjacency,
                             mmr.laplacian,
                             mmr.A3_minus_D])
parser.add_argument("-p", "--path", default="base_from_jaguar",
                    help="Path de la learning base")
parser.add_argument("-t", "--title", default="Sans Titre",
                    help="Curve title")
args = parser.parse_args()

for directory in os.listdir(args.path):
    for representation in args.representation:
        learning_base = mmu.load_base("{0}/{1}".format(args.path, directory))
        learning_base = representation(learning_base)
        # learning_base = mmr.labels_set_to_vec_laplacian_set(learning_base)
        # learning_base = mmr.labels_set_to_vec_adjacency_set(learning_base)
        # learning_base = mmr.learning_base_to_A3_minus_D(learning_base)

        data_set, label_set = mmu.create_sample_label_classification(learning_base)
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

        # X_train, X_test, y_train, y_test = train_test_split(data_set, label_set,
        #                                                     test_size=0.1)
        # clf = svm.SVC()
        # clf = ensemble.RandomForestClassifier()
        # clf = neighbors.KNeighborsRegressor()

        clf = MLPClassifier()
        # clf = tree.DecisionTreeClassifier()
        title = "{0}_{1}".format(directory, representation.__name__)
        mmu.plot_learning_curve(clf, title, data_set, label_set, cv=cv, n_jobs=4)
        plt.savefig("learning_curve/{0}".format(title))
        # pred, miss = mmu.learning(clf, X_train, X_test, y_train, y_test)

        # accu, recal, fmeasure, _ = precision_recall_fscore_support(y_test,
        #                                                            pred,
        #                                                            average="macro")
        # print(directory, representation.__name__)
        # print("|{0}|{1}|{2}%|{3}|{4}|{5}|\n".format(directory,
        #                                             representation.__name__,
        #                                             miss,
        #                                             accu,
        #                                             recal,
        #                                             fmeasure))
