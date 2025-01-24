import pandas as pd
import tensorflow as tf
import sklearn
import scikeras
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)
y = np_utils.to_categorical(y)

def criar_rede():
    k.clear_session()
    rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape = (4,)),
    tf.keras.layers.Dense(units = 4, activation = 'relu'),
    tf.keras.layers.Dense(units = 4, activation = 'relu'),
    tf.keras.layers.Dense(units = 3, activation = 'softmax')
    ])
    rede_neural.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    return rede_neural

rede_neural = KerasClassifier(model = criar_rede, epochs = 250, batch_size = 10)

resultados = cross_val_score(estimator = rede_neural, X = X, y = y, cv = 10, scoring = 'accuracy')

print(resultados)
print(resultados.mean())