import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k

X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

def criar_rede():
  k.clear_session()
  rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape = (30,)),
    tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])
  otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)
  rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
  return rede_neural

rede_neural = KerasClassifier(model = criar_rede, epochs = 100, batch_size = 10)

resultados = cross_val_score(estimator=rede_neural, X=X, y=y.values.ravel(), cv=10, scoring='accuracy')
print(f'Resultados: {resultados}')

print(f'Média: {resultados.mean}')

print(f'Desvio padrão: {resultados.std}')
