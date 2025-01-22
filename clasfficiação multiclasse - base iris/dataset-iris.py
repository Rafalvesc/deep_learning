import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils as np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

base = pd.read_csv('iris.csv')

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

y = np_utils.to_categorical(y)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.25)

rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape = (4,)),
    tf.keras.layers.Dense(units = 4, activation = 'relu'),
    tf.keras.layers.Dense(units = 4, activation = 'relu'),
    tf.keras.layers.Dense(units = 3, activation = 'softmax')
])

rede_neural.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

rede_neural.fit(X_treinamento, y_treinamento, batch_size = 10, epochs = 1000)

rede_neural.evaluate(X_teste, y_teste)

previsoes = rede_neural.predict(X_teste)

previsoes = previsoes > 0.5

y_teste2 = [np.argmax(t) for t in y_teste]
print(y_teste2)

previsoes2 = [np.argmax(t) for t in previsoes]
print(previsoes2)

accuracy= accuracy_score(y_teste2, previsoes2)
print(f"Accuracy Score: {accuracy:.2f}")

confusion_matrix = confusion_matrix(y_teste2, previsoes2)
print(confusion_matrix)