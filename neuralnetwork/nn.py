#neural network
import numpy as np

# Define the activation function (sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input data (X)
# Each row is an input pattern
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Output data (y)
# Each row corresponds to the output for the input pattern
# Example: XOR logic gate outputs
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases randomly
#taking linear regression for the neural network
np.random.seed(1)
weights_input_hidden = np.random.uniform(-1, 1, (2, 2))  # Weights between input and hidden layer
weights_hidden_output = np.random.uniform(-1, 1, (2, 1))  # Weights between hidden and output layer
bias_hidden = np.random.uniform(-1, 1, (1, 2))  # Bias for hidden layer
bias_output = np.random.uniform(-1, 1, (1, 1))  # Bias for output layer

# Learning rate min=0. and max=0.01
learning_rate = 0.1

# Training the neural network
for epoch in range(10000):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_layer_output = sigmoid(final_layer_input)

    # Backpropagation
    output_error = y - final_layer_output
    output_delta = output_error * sigmoid_derivative(final_layer_output)

    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += learning_rate * np.dot(hidden_layer_output.T, output_delta)
    bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta)
    bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

# Test the trained neural network
print("Trained Neural Network Outputs:")
hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
final_layer_output = sigmoid(final_layer_input)

print(final_layer_output)