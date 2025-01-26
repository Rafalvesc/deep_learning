import pandas as pd
import tensorflow as tf
import sklearn
import scikeras
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('iris.csv')

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = np_utils.to_categorical(y)

def criar_rede(optimizer='adam', learning_rate=0.001, activation='relu', hidden_units=(4, 4), **kwargs):
    k.clear_session()
    rede_neural = Sequential()
    rede_neural.add(tf.keras.layers.InputLayer(shape=(4,)))

    for units in hidden_units:
        rede_neural.add(tf.keras.layers.Dense(units=units, activation=activation))
    
    rede_neural.add(tf.keras.layers.Dense(units=3, activation='softmax'))
    
    optimizer_instance = tf.keras.optimizers.get(optimizer)
    optimizer_instance.learning_rate = learning_rate
    
    rede_neural.compile(optimizer=optimizer_instance, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return rede_neural

rede_neural = KerasClassifier(model=criar_rede)

parametros = {
    'model__optimizer': ['adam', 'sgd', 'rmsprop'],
    'model__learning_rate': [0.01, 0.001, 0.0001],
    'model__hidden_units': [(4,), (8,), (4, 4), (8, 4), (8, 8)],
    'model__activation': ['relu', 'tanh'],
    'model__epochs': [50, 100, 250],
    'model__batch_size': [5, 10, 20],
}

grid_search = GridSearchCV(estimator=rede_neural, param_grid=parametros, cv=3, scoring='accuracy')

grid_search.fit(X, y)

print("Melhores parâmetros encontrados:", grid_search.best_params_)
print("Melhor precisão:", grid_search.best_score_)
