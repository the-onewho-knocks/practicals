#python -m venv venv
# venv\Scripts\activate
# pip install numpy matplotlib tensorflow scikit-learn
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pip install notebook
# python -m notebook
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create input values
x = np.linspace(-10, 10, 500)

# Step 2: Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leakyrelu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # numerical stability
    return exp_x / np.sum(exp_x)

# Step 3: Compute outputs
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky = leakyrelu(x)
y_softmax = softmax(x)

# Step 4: Plot graphs
plt.figure(figsize=(12, 7), dpi=120)

plt.plot(x, y_sigmoid, label="Sigmoid", linewidth=2)
plt.plot(x, y_tanh, label="Tanh", linewidth=2)
plt.plot(x, y_relu, label="ReLU", linewidth=2)
plt.plot(x, y_leaky, label="Leaky ReLU", linewidth=2)

# Softmax plotted separately style (dashed)
plt.plot(x, y_softmax, label="Softmax (distribution)", linewidth=2, linestyle='--')

# Axes lines at center
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

# Labels & title
plt.title("Activation Functions (with Softmax)", fontsize=16)
plt.xlabel("Input (x)", fontsize=12)
plt.ylabel("Output", fontsize=12)

# Limits
plt.xlim(-10, 10)
plt.ylim(-1.5, 10)

# Grid
plt.grid(True, linestyle='--', alpha=0.6)

# Legend
plt.legend()

# Show plot
plt.show()