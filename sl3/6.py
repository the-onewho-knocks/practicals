# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    multilabel_confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.options.display.max_columns = None

# =========================
# LOAD DATASET
# =========================
iris = pd.read_csv("Iris.csv")

# Drop Id column if exists
if 'Id' in iris.columns:
    iris = iris.drop(columns=['Id'])

print("Head:\n", iris.head())
print("Tail:\n", iris.tail())

# =========================
# BASIC INFO
# =========================
print("Shape:", iris.shape)
print("Size:", iris.size)

print("\nInfo:")
iris.info()

print("\nDescribe:")
print(iris.describe())

print("\nClass distribution:")
print(iris['Species'].value_counts())

# =========================
# INSERT NULL VALUES (FOR PRACTICE)
# =========================
null_rows = pd.DataFrame([
    {'SepalLengthCm': 6.0, 'SepalWidthCm': 3.2, 'PetalLengthCm': np.nan, 'PetalWidthCm': 2.0, 'Species': 'Iris-virginica'},
    {'SepalLengthCm': np.nan, 'SepalWidthCm': 2.9, 'PetalLengthCm': 4.5, 'PetalWidthCm': np.nan, 'Species': 'Iris-versicolor'},
    {'SepalLengthCm': 14.0, 'SepalWidthCm': 2.9, 'PetalLengthCm': 4.5, 'PetalWidthCm': np.nan, 'Species': np.nan}
])

insert_position = 40
iris_with_null = pd.concat([iris.iloc[:insert_position], null_rows, iris.iloc[insert_position:]])

print("\nNull counts before removal:")
print(iris_with_null.isnull().sum())

# =========================
# REMOVE NULL VALUES
# =========================
iris_no_null = iris_with_null.dropna().reset_index(drop=True)

print("\nShape after dropna:", iris_no_null.shape)
print("Null counts after dropna:\n", iris_no_null.isnull().sum())

# =========================
# VISUALIZATION
# =========================
numeric_cols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

# Pairplot
sns.pairplot(iris_no_null, hue='Species')
plt.show()

# Distribution
iris_no_null[numeric_cols].hist(bins=12, figsize=(10,6))
plt.show()

# Heatmap
sns.heatmap(iris_no_null[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================
# OUTLIER REMOVAL (IQR)
# =========================
iris_clean = iris_no_null.copy()

for col in numeric_cols:
    q1 = iris_clean[col].quantile(0.25)
    q3 = iris_clean[col].quantile(0.75)
    iqr = q3 - q1
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    iris_clean = iris_clean[(iris_clean[col] >= lower) & (iris_clean[col] <= upper)]

print("Shape after outlier removal:", iris_clean.shape)

sns.boxplot(data=iris_clean[numeric_cols])
plt.title("Boxplot after outlier removal")
plt.show()

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
iris_clean['Species_num'] = le.fit_transform(iris_clean['Species'])

print("Species mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# =========================
# FEATURES & LABELS
# =========================
X = iris_clean[numeric_cols]
y = iris_clean['Species_num']

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFirst 5 scaled rows:")
print(pd.DataFrame(X_scaled, columns=numeric_cols).head())

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# =========================
# NAIVE BAYES MODEL
# =========================
nb = GaussianNB()
nb.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
y_pred = nb.predict(X_test)

print("\nPredicted:", y_pred[:10])
print("Actual   :", y_test.values[:10])

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# METRICS
# =========================
mcm = multilabel_confusion_matrix(y_test, y_pred)

for idx, label in enumerate(le.classes_):
    tn, fp, fn, tp = mcm[idx].ravel()
    print(f"Class {label}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))