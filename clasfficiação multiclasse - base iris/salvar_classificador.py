import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

base = pd.read_csv('iris.csv')

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = np_utils.to_categorical(y)

# Criar o modelo
classificador = Sequential()
classificador.add(InputLayer(input_shape=(4,)))
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=3, activation='softmax'))

classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

classificador.fit(X, y, batch_size=4, epochs=300)

novo = np.array([[1, 0, 0, 0]])

previsao = classificador.predict(novo)

print("Previsão:", previsao)

classe_prevista = np.argmax(previsao, axis=1)
print("Classe prevista:", label_encoder.inverse_transform(classe_prevista))

classificador.save('classificador_iris.keras')
