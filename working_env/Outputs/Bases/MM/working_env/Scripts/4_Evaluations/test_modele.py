from sklearn.externals import joblib
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from minerminor import mm_draw as mmd
from sklearn.model_selection import train_test_split
import os
from os.path import isfile, join
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Path to classifier to test
mypath = "classifier/class_test"
# Base to use for testing
base_path = "bases/base_tw1_rdm_test"
representation = [mmr.adjacency, mmr.laplacian]
res = {}
resultat_file = open("resultats/log_testing.txt", "w")
resultat_file.write("Base|Representation|Score|Mat_conf|Vecteur_res\n")
for clf_file in [f for f in os.listdir(mypath) if isfile(join(mypath, f))]:
    clf = joblib.load(mypath+"/"+clf_file)
    arr_info_clf = clf_file.split("_")
    res[arr_info_clf[0]] = {}
    for base_dir in os.listdir(base_path):
        for rep in representation:
            learning_base = mmu.load_base(base_path+"/"+base_dir)
            arr_info_base = base_dir.split("_")
            arr_class = arr_info_base[2][1:-1].split(",")
            # mmu.show_graph(learning_base[1][0])
            learning_base = rep(learning_base)

            data_set, label_set = mmu.create_sample_label_classification(learning_base)
            for count, label in enumerate(label_set):
                if label > 0:
                    label_set[count] = 1
            # print(label_set)
            # X_train, X_test, y_train, y_test = train_test_split(data_set, label_set,
            #                                                     test_size=0.1)

            # Classification on data test .
            y_pred = clf.predict(data_set)
            classified = (label_set == y_pred).sum()
            total = len(y_pred)

            # score = cross_val_score(clf, data_set, label_set, cv=10)
            mat_conf = confusion_matrix(label_set, y_pred)
            report = classification_report(label_set, y_pred, target_names=['P', '!P'])
            # y_pred = cross_val_predict(clf, data_set, label_set, cv=4)
            # print(len(y_pred))
            # mat_conf_aug = confusion_matrix(label_set, y_pred)

            # print("Number of mislabeled on a total %d : %d (%d %%)" % (
            #     total, miss, (miss/total)*100))
            res[arr_info_clf[0]][int(arr_class[1])] = classified
            resultat_file.write("{0}|{1}|\n{2}|\n{3}|\n{4}|\n".format(base_dir,
                                                                       rep.__name__,
                                                                       clf,
                                                                       mat_conf.tolist(),
                                                                       report))
print(res)
# mmd.create_curve_xP(res)
# générer la data
# draw la data
