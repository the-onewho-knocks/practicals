# =========================
# 1. IMPORT LIBRARIES
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# =========================
# 2. LOAD DATASET
# =========================
# (Make sure Automobile.csv is in the same folder)
df = pd.read_csv("Automobile.csv")
df.head()

# =========================
# 3. DATA UNDERSTANDING
# =========================
print("\nStatistical Summary:")
print(df.describe())

print("\nShape of dataset:", df.shape)
print("Total elements:", df.size)

# =========================
# 4. CHECK MISSING VALUES
# =========================
print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# 5. HANDLE MISSING VALUES
# =========================
# Fill numeric columns with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nAfter handling missing values:")
print(df.isnull().sum())

# =========================
# 6. DATA TYPES
# =========================
print("\nData Types:")
print(df.dtypes)

# =========================
# 7. DATA TYPE CONVERSION
# =========================
# Convert horsepower to numeric if needed
if 'horsepower' in df.columns:
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# =========================
# 8. NORMALIZATION (Min-Max)
# =========================
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

print("\nAfter Normalization:")
print(df.head())

# =========================
# 9. ENCODING CATEGORICAL DATA
# =========================
df_encoded = pd.get_dummies(df)

print("\nAfter Encoding:")
print(df_encoded.head())

# =========================
# 10. VISUALIZATION
# =========================

# Histogram
df_encoded.hist(figsize=(10, 8))
plt.suptitle("Data Distribution")
plt.show()

# Heatmap (only numeric columns)
plt.figure(figsize=(10, 8))
numeric_df = df_encoded.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()