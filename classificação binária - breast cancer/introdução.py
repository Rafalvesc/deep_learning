import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix

x = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.25)

rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape = (30,)),
    tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
    tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
])

otimizador = tf.keras.optimizers.Adam(learning_rate = 0.01, clipvalue = 0.5)

rede_neural.compile(otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

rede_neural.fit(x_treinamento, y_treinamento, batch_size = 10, epochs = 100)

pesos0 = rede_neural.layers[0].get_weights()
pesos1 = rede_neural.layers[1].get_weights()
pesos2 = rede_neural.layers[2].get_weights()

previsoes = rede_neural.predict(x_teste)
previsoes = previsoes > 0.5

accuracy_score(y_teste, previsoes)
confusion_matrix(y_teste, previsoes)

print(accuracy_score(y_teste,previsoes))
print(confusion_matrix(y_teste, previsoes))

rede_neural.evaluate(x_teste, y_teste)

print(rede_neural.evaluate(x_teste, y_teste))