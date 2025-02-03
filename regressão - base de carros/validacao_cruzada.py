import pandas as pd
import tensorflow as tf
import sklearn
import scikeras
import time

from scikeras.wrappers import KerasRegressor
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics

inicio = time.time()

base = pd.read_csv('autos.csv', encoding='ISO-8859-1')

base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
base = base.drop('name', axis=1)
base = base.drop('seller', axis=1)
base = base.drop('offerType', axis=1)

base = base[base.price > 10]
base = base.loc[base.price < 350000]

valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell',
           'model': 'golf',
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value=valores)

X = base.iloc[:, 1:12].values
y = base.iloc[:, 0].values

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])], remainder='passthrough')
X = onehotencoder.fit_transform(X).toarray()

def criar_rede():
  k.clear_session()
  regressor = Sequential([
    tf.keras.layers.InputLayer(shape=(316,)),
    tf.keras.layers.Dense(units=158, activation='relu'),
    tf.keras.layers.Dense(units=158, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')])
  regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
  return regressor

regressor = KerasRegressor(model = criar_rede, epochs = 100, batch_size = 300)

resultados = cross_val_score(estimator=regressor, X = X, y = y,cv=5, scoring='neg_mean_absolute_error')

fim= time.time()
print(fim-inicio/60/60)

print(abs(resultados))

print(f'Média: {abs(resultados.mean())}')