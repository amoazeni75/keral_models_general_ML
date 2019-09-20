# -*- coding: utf-8 -*-
"""
@author: S.Alireza Moazeni(S.A.M.P.8)
@tutorial source: Deep Learning With Python, Develop Deep Learning Models On Theano And TensorFlow Using
Keras, Jason Brownlee
"""

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init= 'uniform' , activation= 'relu' ))
    model.add(Dense(8, init= 'uniform' , activation= 'relu' ))
    model.add(Dense(1, init= 'uniform' , activation= 'sigmoid' ))
    # Compile model
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model

def create_model_grid(optimizer= 'rmsprop' , init= 'glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init=init, activation= 'relu'))
    model.add(Dense(8, init=init, activation= 'relu'))
    model.add(Dense(1, init=init, activation= 'sigmoid' ))
    # Compile model
    model.compile(loss= 'binary_crossentropy' , optimizer=optimizer, metrics=[ 'accuracy' ])
    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
"""
In this section we must pass call functio  to KerasClassifier by build_fn argument
turn off output of every epoch by setting verbos=0
"""
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10, verbose=0)

# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
"""
We use the scikit-learn function cross val score() to evaluate our model using the 
cross validation scheme and print the results.
This function will train our network.
"""
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())







# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# grid search epochs, batch size and optimizer
optimizers = [ 'rmsprop' , 'adam']
init = [ 'glorot_uniform' , 'normal' , 'uniform' ]
epochs = numpy.array([50, 100, 150])
batches = numpy.array([5, 10, 20])
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
