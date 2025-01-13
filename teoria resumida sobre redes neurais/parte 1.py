import numpy as np

# Função degrau (step function): usada para ativação binária, retorna 1 se a entrada é maior ou igual a 1, caso contrário retorna 0.
def step_function(soma):
    if soma >= 1:
        return 1
    return 0

# Função sigmoide: transforma a entrada em um valor entre 0 e 1, sendo útil para modelagem probabilística.
def sigmoid_function(soma):
    return 1 / (1 + np.exp(-soma))

# Função tangente hiperbólica (tanh): similar à sigmoide, mas retorna valores entre -1 e 1, útil para normalizar saídas.
def tahn_function(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

# Função ReLU (Rectified Linear Unit): retorna a entrada se for maior ou igual a 0, caso contrário retorna 0. É amplamente usada por sua simplicidade e eficiência em redes profundas.
def relu_function(soma):
    if soma >= 0:
        return soma
    return 0

# Função Softmax: converte uma lista de valores em probabilidades, usada geralmente na camada de saída para classificação.
def softmax_function(x):
    ex = np.exp(x)
    return ex / ex.sum()

# Função linear: retorna a entrada como está, usada em regressão ou em camadas intermediárias.
def linear_function(soma):
    return soma
print(linear_function(2.1))
