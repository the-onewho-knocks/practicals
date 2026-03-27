# Import libraries
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
# Load dataset
digits = load_digits()
# Features and labels
X = digits.data
y = digits.target
# Convert to even (0) and odd (1)
y = np.where(y % 2 == 0, 0, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Train perceptron model
model = Perceptron(max_iter=1000) #this creates a model which learns 1000
model.fit(X_train, y_train) #this is were the learning of model takes place

# Predictions
y_pred = model.predict(X_test) #if score >= 0 → 1 (Odd) else → 0 (Even)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred)) # comparison between predicted values vs actual values 
# Confusion Matrix

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Show some predictions
for i in range(5):
    plt.imshow(X_test[i].reshape(8,8), cmap='gray')
    plt.title(f"Actual: {y_test[i]} Predicted: {y_pred[i]}")
    plt.show()