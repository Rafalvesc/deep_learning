import pandas as pd
import tensorflow as tf
import sklearn
import scikeras


from scikeras.wrappers import KerasClassifier
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV

X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):
  k.clear_session()
  rede_neural = Sequential([
      tf.keras.layers.InputLayer(shape=(30,)),
      tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dropout(rate = 0.2),
      tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dropout(rate = 0.2),
      tf.keras.layers.Dense(units=1, activation = 'sigmoid')])
  rede_neural.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
  return rede_neural

rede_neural = KerasClassifier(model = criar_rede)

parametros = {
    'batch_size': [10, 30],
    'epochs': [50],
    'model__optimizer': ['adam'],
    'model__loss': ['binary_crossentropy'],
    'model__kernel_initializer': ['random_uniform', 'normal'],
    'model__activation': ['relu'],
    'model__neurons': [16]
}

grid_search = GridSearchCV(estimator = rede_neural, param_grid = parametros,
    scoring = 'accuracy', cv = 5)

grid_search = grid_search.fit(X, y)

print(grid_search)

melhores_parametros = grid_search.best_params_
print(melhores_parametros)

melhor_precisao = grid_search.best_score_
print(melhor_precisao)