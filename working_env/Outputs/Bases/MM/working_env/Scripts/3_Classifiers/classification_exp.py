"""Test tree classification."""
import networkx as nx
import numpy as np
from minerminor import mm_generator as mmg
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from minerminor import mm_draw as mmd
from sklearn import ensemble, svm, neighbors, tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import os
import argparse
import datetime as dt
from sklearn.model_selection import ShuffleSplit
from sklearn.externals import joblib

parser = argparse.ArgumentParser(prog="Script de classification")
parser.add_argument("-r", "--representation", nargs='*',
                    default=[mmr.adjacency,
                             mmr.laplacian])
parser.add_argument("-c", "--classifieur", nargs='*',
                    default=[tree.DecisionTreeClassifier, svm.SVC, MLPClassifier])
parser.add_argument("-p", "--path", default="base_from_jaguar",
                    help="Path de la learning base")
parser.add_argument("-s", "--save", action='store_true', default=False, help="Save the classifieur")

args = parser.parse_args()

resultat_file = open("resultats/clf_log.txt".format(args.path, dt.datetime.now().time()), "a")

for directory in os.listdir(args.path):
    for classifieur in args.classifieur:
        for representation in args.representation:
            learning_base = mmu.load_base("{0}/{1}".format(args.path, directory))
            learning_base = representation(learning_base)

            data_set, label_set = mmu.create_sample_label_classification(learning_base)
            X_train, X_test, y_train, y_test = train_test_split(data_set, label_set,
                                                                test_size=0.2)
            clf = classifieur()
            cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
            title = '{0}_{1}_{2}'.format(classifieur.__name__, representation.__name__, len(label_set))
            # mmd.plot_learning_curve(clf, title, data_set, label_set, ylim=(0., 1.01), cv=cv, n_jobs=4)
            pred, miss, clf = mmu.learning(clf, X_train, X_test, y_train, y_test)

            score = cross_val_score(clf, X_test, y_test, cv=10)
            mat_conf = confusion_matrix(y_test, pred)

            accu, recal, fmeasure, _ = precision_recall_fscore_support(y_test,
                                                                       pred, average='macro')
            to_save = "{0}|{1}|{2}|{3}%|{4}|{5}|{6}|{7}|{8}\n".format(directory.replace("_", "|"),
                                                                      classifieur.__name__,
                                                                      representation.__name__,
                                                                      miss,
                                                                      score.mean(),
                                                                      recal,
                                                                      fmeasure,
                                                                      mat_conf.tolist(),
                                                                      score.tolist())
            print(directory, representation.__name__)
            print(to_save)
            resultat_file.write(to_save)
            if args.save:
                joblib.dump(clf, 'classifier/{0}_{1}_{2}_{3}.pkl'.format(classifieur.__name__, representation.__name__, len(label_set), directory))
