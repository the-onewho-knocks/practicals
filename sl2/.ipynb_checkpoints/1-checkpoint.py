import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create input values
x = np.linspace(-10, 10, 100)

# Step 2: Define activation functions

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh
def tanh(x):
    return np.tanh(x)

# ReLU
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU
def leakyrelu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stability fix
    return exp_x / np.sum(exp_x)

# Step 3: Compute outputs
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky = leakyrelu(x)

# Step 4: Plot graphs
plt.figure(figsize=(10, 6))

plt.plot(x, y_sigmoid, label="Sigmoid")
plt.plot(x, y_tanh, label="Tanh")
plt.plot(x, y_relu, label="ReLU")
plt.plot(x, y_leaky, label="Leaky ReLU")

plt.title("Activation Functions in Neural Networks")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid()

# Show plot without blocking
plt.show(block=False)
plt.pause(5)   # display for 5 seconds
plt.close()