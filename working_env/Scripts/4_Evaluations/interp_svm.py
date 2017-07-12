"""Interpret SVM."""
import numpy as np
from sklearn.externals import joblib
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


clf = joblib.load("working_env/classifier/SVC_laplacian_2000.pkl")
learning_base = mmu.load_base("working_env/base_validation/pTree-basic-cycle-generation_14_[0, 1]_1000")
learning_base = mmr.A3_minus_D(learning_base)

data_set, label_set = mmu.create_sample_label_classification(learning_base)
X_train, X_test, y_train, y_test = train_test_split(data_set, label_set, test_size=0.1)

pca = PCA(n_components=2).fit(X_train)
pca_2d = pca.transform(X_train)

for i in range(0, pca_2d.shape[0]):
    if y_train[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r',    marker='+')
    elif y_train[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g',    marker='o')

plt.legend([c1, c2], ["0-P", "1-P"])
x_min, x_max = pca_2d[:, 0].min() - 1,   pca_2d[:, 0].max() + 1
y_min, y_max = pca_2d[:, 1].min() - 1,   pca_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
Z = clf.decision_functions(np.c_[xx.ravel(),  yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z)
plt.axis('off')
plt.show()

# plt.figure(1)
# plt.clf()
# print(clf.n_support_[:])
# print(clf.support_[:])
# print(clf.support_vectors_[:, 0])
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10, cmap=plt.cm.Paired)
# plt.show()
