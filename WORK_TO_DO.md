MAJOR ADVANCES (important things that we will try to do first):

- Planarity node2vec matrixes: 

When we generate several time node2vec bases with same parameters, we obtain different matrixes what led to significant different f measure

We have to set properly node2vec to rectify this and stabilize our matrixes.

Our intuition : set properly the length of walks

Note : We did not have this problem with TW1 node2vec bases and TW2 node2vec bases

- Hyperparameters: 

We're actually doing an exhaustive search for node2vec (p, q, size) hyperparameters 

We should find ways to exclude specific values (fix some values then change the others and try to find a law for changed values ..)

Try to optimize values which have a predictable behavior

MINOR CHANGES

- Optimize node2vec matrixes generation time: 

Take a close look at node2vec documentation and watch if changing p and q involve to alter the entire matrix
Maybe we can just store the matrixes in memory (not saving them like we are doing atm) and change specific values

- Learning curve

In order to find best base size we have to generate and analyze learning curves.

We had implemented learning curve in a previous version of our work but we need to update it to our current work.

http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html and mm_draw and others implemented files

- Graphic changes

Display a bar or a timer to see node2vec matrixes progression
