import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create input values
x = np.linspace(-10, 10, 100)

# Step 2: Define activation functions

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  #0 to 1

# Tanh
def tanh(x):
    return np.tanh(x)   #-1 to 1

# ReLU
def relu(x):
    return np.maximum(0, x)

# Step 3: Compute outputs
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

# Step 4: Plot graphs

plt.figure(figsize=(10, 6))

plt.plot(x, y_sigmoid, label="Sigmoid", color='blue')
plt.plot(x, y_tanh, label="Tanh", color='green')
plt.plot(x, y_relu, label="ReLU", color='red')

plt.title("Activation Functions in Neural Networks")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid()

plt.show()