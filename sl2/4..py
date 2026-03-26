# =========================
# "This code implements a Perceptron classifier for digit recognition.
#  The dataset contains 15-dimensional binary features representing 5×3 digit images. PCA is applied to reduce dimensionality to 2D for visualization. 
# The perceptron learns a linear decision boundary using weight updates on misclassification.
#  Finally, decision regions are plotted to visualize how the model separates digit 0 from other digits.
# =========================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================
# DATASET (5x3 digits → 15 features)
# =========================
X = np.array([
    [1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1],  # 0
    [0,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,1],  # 1
    [1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1],  # 2
    [1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1],  # 3
    [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1],  # 4
    [1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1],  # 5
    [1,1,1, 1,0,0, 1,1,1, 1,0,1, 1,1,1],  # 6
    [1,1,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1],  # 7
    [1,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1],  # 8
    [1,1,1, 1,0,1, 1,1,1, 0,0,1, 1,1,1]   # 9
])

# =========================
# LABELS (0 vs rest)
# =========================
y = np.where(np.arange(10) == 0, 1, -1)  

# =========================
# PCA → reduce to 2D  pca= principle component analysis
# =========================
pca = PCA(n_components=2)
X = pca.fit_transform(X)  #the less important data is reduced and important patterns are kept

# =========================
# INITIALIZE PARAMETERS
# =========================
w = np.zeros(X.shape[1]) #weight
b = 0   #bias

# =========================
# PERCEPTRON FUNCTION
# =========================
def perceptron(X, y, w, b, lr=0.1, epochs=100): #lr is learning rate
    for _ in range(epochs):
        for i in range(len(X)):
            z = np.dot(X[i], w) + b
            y_pred = 1 if z >= 0 else -1

            if y_pred != y[i]: #if prediction is wrong then update the values of weight and bias
                w = w + lr * y[i] * X[i]
                b = b + lr * y[i]

    return w, b

# =========================
# TRAIN MODEL
# =========================
w, b = perceptron(X, y, w, b)

# =========================
# DECISION REGION
# =========================
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(  #this creates a grid of points in 2d space
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1)
)

Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b) #converts 2d grid into 1d grid and combines both into coordinate pairs
Z = Z.reshape(xx.shape)

# =========================
# PLOT
# =========================
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=100, edgecolors='k')

plt.title("Perceptron Decision Region (Digit 0 vs Rest)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()