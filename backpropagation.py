import numpy as np

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Dados de entrada e saída
X = np.array([[0], [1]])
y = np.array([[0], [1]])

# Inicialização de pesos e bias com valores aleatórios
np.random.seed(42)
weights_1 = np.random.rand(1, 2)  # da entrada para camada oculta (1x2)
bias_1 = np.random.rand(1, 2)     # bias da camada oculta (1x2)
weights_2 = np.random.rand(2, 1)  # da camada oculta para saída (2x1)
bias_2 = np.random.rand(1, 1)     # bias da saída

# Taxa de aprendizado
lr = 0.1

# Treinamento
for epoch in range(100000):
    # FORWARD PASS
    hidden_input = np.dot(X, weights_1) + bias_1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_2) + bias_2
    final_output = sigmoid(final_input)

    # ERRO (custo)
    error = y - final_output
    cost = np.mean(np.square(error))  # MSE

    if epoch % 1000 == 0:
        print(f"Época {epoch} - Custo: {cost:.4f}")

    # BACKPROPAGATION
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(weights_2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # AJUSTE DE PESOS E BIAS
    weights_2 += hidden_output.T.dot(d_output) * lr
    bias_2 += np.sum(d_output, axis=0, keepdims=True) * lr

    weights_1 += X.T.dot(d_hidden) * lr
    bias_1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

# Teste
print("\nSaídas finais da rede:")
print(final_output)
