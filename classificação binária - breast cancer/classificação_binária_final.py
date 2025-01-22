import pandas as pd
import tensorflow as tf
import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dropout


X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

def criar_rede():
  k.clear_session()
  rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape = (30,)),
    tf.keras.layers.Dense(units = 22, activation = 'relu', kernel_initializer = 'random_uniform'),
    Dropout(0.2),
    tf.keras.layers.Dense(units = 20, activation = 'relu', kernel_initializer = 'random_uniform'),
    Dropout(0.2),
    tf.keras.layers.Dense(units = 20, activation = 'relu', kernel_initializer = 'random_uniform'),
    Dropout(0.2),
    tf.keras.layers.Dense(units = 20, activation = 'relu', kernel_initializer = 'random_uniform'),
    Dropout(0.2),
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])
  otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)
  rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
  return rede_neural

rede_neural = KerasClassifier(model = criar_rede, epochs = 100, batch_size = 46)

resultados = cross_val_score(estimator=rede_neural, X=X, y=y.values.ravel(), cv=10, scoring='accuracy')
print(f'Resultados das 10 validações: {resultados}')
print(f'Média da acurácia: {np.mean(resultados):.4f}')
print(f'Desvio padrão da acurácia: {np.std(resultados):.4f}')

