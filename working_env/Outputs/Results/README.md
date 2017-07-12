# Chap 1 : Workflow -> Protocole Expérimental  |  Utilité

# Chap 2 : Des Représentations.

# Chap 3 : Des types de classifications.

# Chap 4 : De la construction des bases d'apprentissage à la validation.

# Chap 5 : De la mise à l'échelle

# Chap 6 : Conclusion & Perspective


## Résumé

> Deux révolution : machine learning (RL + CNN ) & graph minor = > unir les deux pour estimer des props dans les graphes

> Workflow : I (input) -> B (base) -> R (représentation) -> C (classifier) -> E/V (evaluation & validation)

> Représentation : Adjacency, Laplacienne, (PCA, QR, RW) ?

> Classifier : DT, SVM, MLP, CNN

> List des prop client à tester : TW, PW, BW, Planar, CW

> Protocole de génération de modèle (1 par Représentation) : Bi + CV, Bi + rdm, rdm + rdm, Bi | rdm + rdm (Bi for CPU, rdm for robust)

> Génération des Bi via extremal graphes génération (TW1, TW2, TW3, Planar)
> Certificat des propriétés (Reduction rules -> TW, Certificat simple -> planar) => Génération des bases rdm

> Validation : F1_score, matrice confusion, learning curves

> Scalabilité : Naive method and percpective

> EXP WILL NEED TO BE DONE : TW1(0p-1p), TW2(extremal + Reduction rules), TW3 (?), TW3 + Planar

> Attendre les refs de Eric pour les méthodes sur TW
