"""CNN for planar property recognition."""
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from keras.utils import plot_model
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from minerminor import mm_draw as mmd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
import networkx as nx
import numpy as np
import keras
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping



def data_format(learning_base):
    # datarows formating
    # -- extract from learning base et format it
    data_set, label_set = mmu.create_sample_label_classification(learning_base)
    # --reshape for cnn
    data_set = np.array(data_set)
    data_set = data_set.reshape(data_set.shape[0], graph_dim, graph_dim, 1)
    x_train, x_test, y_train, y_test = train_test_split(data_set, label_set, test_size=0.2)
    # -- list to np.array
    x_train, x_test, y_train, y_test = [np.array(i) for i in [x_train, x_test, y_train, y_test]]

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, label_set


def nn_model():
    """Model v0.1."""
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


def cnn_model_alpha():
    """Model v0.1."""
    model = Sequential()
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2), input_shape=input_shape))
    model.add(Conv2D(36, 5, 5, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


def cnn_model_test():
    """Init a Keras modele to be warp to scilearn."""
    # model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


# hyperparameters
init_model = nn_model
n2v = False
save = False
batch_size = 128
num_classes = 2
epochs = 200
title = "clf_cnn_planar_rdm_n2v"
base_path = "bases/base_planar_rdm_test/learning-base-planar_18_[0, 1]_1000"
base_test_path = "bases/base_planar_rdm_test/learning-base-planar_18_[0, 1]_1000"
graph_dim = 18
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
# rep_1 = lambda x: nx.to_numpy_matrix(x)
rep_1 = lambda x: nx.laplacian_matrix(x).toarray()
rep_2 = lambda x: mmr.mat_to_PCA(x)
rep_arr = [rep_1, rep_2]
# -- declaration input_shape (lis en global par la fonction de creation model... pas beau !)
input_shape = (graph_dim, graph_dim, 1)


# load data
if not n2v:
    learning_base = mmu.load_base(base_path)
    # representation
    learning_base = mmr.learning_base_to_rep(learning_base, rep_arr)
else:
    learning_base = mmu.load_base_n2v(base_path)

x_train, x_test, y_train, y_test, _ = data_format(learning_base)

# model init
# model = init_model()
model = KerasClassifier(build_fn=init_model, epochs=epochs, batch_size=batch_size, verbose=0)

# model fitting
history = model.fit(x_train, y_train,
                    epochs=epochs, batch_size=batch_size, verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping])

if not os.path.exists('resultats/'+title):
    os.makedirs('resultats/'+title)

if save:
    model.save('classifier/'+title+'.h5')

# load data
if not n2v:
    learning_base = mmu.load_base(base_path)
    # representation
    learning_base = mmr.learning_base_to_rep(learning_base, rep_arr)
else:
    learning_base = mmu.load_base_n2v(base_path)

x_train, x_test, y_train, y_test, label_set = data_format(learning_base)
# Visualisation
# plot_model(model, to_file='resultats/curve_pool/model_schema.png')
data_set = np.concatenate([x_train, x_test])
y_pred = model.predict(data_set)

mat_conf = confusion_matrix(label_set, y_pred)
report = classification_report(label_set, y_pred, target_names=['P', '!P'])

print("{0}|{1}|\n\n{2}|\n{3}|\n{4}".format(base_path,
                                           rep_arr,
                                           model,
                                           mat_conf.tolist(),
                                           report))

mmd.plot_acc_hist_keras(history)
mmd.plot_loose_hist_keras(history)
# model.summary()
# score = model.evaluate(x_test, y_test, verbose=1)
# print('\nTest loss:', score[0])
# print('Test accuracy:', score[1])
