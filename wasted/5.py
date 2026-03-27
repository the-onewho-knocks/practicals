# =========================
# IMPORT LIBRARIES
# =========================
import numpy as np

# =========================
# ACTIVATION FUNCTIONS
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# =========================
# DATASET (XOR)
# =========================
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

#y is the actual expected output
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# =========================
# INITIALIZE PARAMETERS
# =========================
np.random.seed(42)

input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# Weights and Biases
W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
b1 = np.random.uniform(size=(1, hidden_neurons))

W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
b2 = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.1
epochs = 10000

# =========================
# TRAINING (FORWARD + BACKPROP)
# =========================
for epoch in range(epochs):

    # ---- Forward Propagation ----
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(final_input)

    # ---- Error Calculation ----
    error = y - predicted_output

    # ---- Backpropagation ----
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = np.dot(d_output, W2.T) #.T means transpose of rows and columns
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # ---- Update Weights & Biases ----
    W2 += np.dot(hidden_output.T, d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate   #axis=0 means summation will be column wise
    #keepdims keeps the result as 2d array

    W1 += np.dot(X.T, d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# =========================
# FINAL OUTPUT
# =========================
print("Final Predicted Output:")
print(predicted_output)