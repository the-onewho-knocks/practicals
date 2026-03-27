import numpy as np

# ------------------------------
# Step 1: Dataset (5x3 digits → 15 features)
# ------------------------------
X = np.array([
    [1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1], # 0
    [0,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,1], # 1
    [1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1], # 2
    [1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1], # 3
    [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1], # 4
    [1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1], # 5
    [1,1,1, 1,0,0, 1,1,1, 1,0,1, 1,1,1], # 6
    [1,1,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1], # 7
    [1,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1], # 8
    [1,1,1, 1,0,1, 1,1,1, 0,0,1, 1,1,1]  # 9
])

# Labels (One-hot encoding) it creates an identity matrix
y = np.eye(10)

# ------------------------------
# Step 2: Initialize parameters
# ------------------------------
input_size = 15
output_size = 10

np.random.seed(0)
weights = np.random.randn(input_size, output_size)
bias = np.zeros((1, output_size))

learning_rate = 0.1
epochs = 1000

# ------------------------------
# Step 3: Softmax function
# ------------------------------
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ------------------------------
# Step 4: Training
# ------------------------------
for epoch in range(epochs):
    z = np.dot(X, weights) + bias
    y_pred = softmax(z)

    error = y - y_pred

    weights += learning_rate * np.dot(X.T, error)
    bias += learning_rate * np.sum(error, axis=0, keepdims=True)

# ------------------------------
# Step 5: Recognition function
# ------------------------------
def recognize_digit(test_input):
    z = np.dot(test_input, weights) + bias
    y_pred = softmax(z)

    confidence = np.max(y_pred)
    predicted_digit = np.argmax(y_pred)

    print("Predicted Digit:", predicted_digit)
    print("Confidence:", round(confidence, 3))

    if confidence > 0.8:
        print("Output: 1 (Recognized)")
        return 1
    else:
        print("Output: 0 (Not Recognized)")
        return 0

# ------------------------------
# Step 6: Test examples
# ------------------------------
print("Test with known digit (3):")
recognize_digit(X[3].reshape(1, -1))

print("\nTest with unknown pattern:")
unknown = np.array([[0,0,0, 0,1,0, 0,1,0, 0,1,0, 0,0,0]])
recognize_digit(unknown)