from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Carregar os dados
base = pd.read_csv('autos.csv', encoding='ISO-8859-1')
base = base.drop(['dateCrawled', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen', 'name', 'seller', 'offerType'], axis=1)
base = base[base.price > 10]
base = base.loc[base.price < 350000]
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 'fuelType': 'benzin', 'notRepairedDamage': 'nein'}
base = base.fillna(value=valores)

X = base.iloc[:, 1:12].values
y = base.iloc[:, 0].values

# Pré-processamento com OneHotEncoder
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])], remainder='passthrough')
X = onehotencoder.fit_transform(X).toarray()

# Função para criar o modelo da rede neural
def criar_rede(loss_function='mean_absolute_error'):
    k.clear_session()
    regressor = Sequential([
        tf.keras.layers.InputLayer(shape=(316,)),
        tf.keras.layers.Dense(units=158, activation='relu'),
        tf.keras.layers.Dense(units=158, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    regressor.compile(loss=loss_function, optimizer='adam', metrics=['mean_absolute_error'])
    return regressor

# Wrap do modelo Keras
regressor = KerasRegressor(model=criar_rede, epochs=20, batch_size=300)

# Definir a grade de hiperparâmetros para a busca
param_grid = {
    'model__loss_function': [
        'mean_squared_error', 
        'mean_absolute_error', 
        'mean_absolute_percentage_error', 
        'mean_squared_logarithmic_error', 
        'squared_hinge'
    ]
}

# Usar o GridSearchCV para encontrar a melhor função de erro
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')
inicio = time.time()

# Realizar a busca
grid_search.fit(X, y)

fim = time.time()

# Resultados da busca
print(f'Tempo total de execução: {(fim - inicio) / 60:.2f} minutos')
print(f'Melhor per'da encontrada: {grid_search.best_score_}')
print(f'Melhor função de erro: {grid_search.best_params_}')
