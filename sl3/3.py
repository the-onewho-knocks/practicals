# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# LOAD ADULT DATASET
# =========================
df = pd.read_csv("adults.csv", skipinitialspace=True)

# Clean column names
df.columns = df.columns.str.strip()

# Rename columns properly
df.columns = [
    "age","workclass","fnlwgt","education","education_num",
    "marital_status","occupation","relationship","race","sex",
    "capital_gain","capital_loss","hours_per_week","native_country","income"
]

# =========================
# HANDLE MISSING VALUES
# =========================
df.replace('?', np.nan, inplace=True)

# Fill categorical missing values with mode
df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])

# Fill numeric missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# =========================
# DISPLAY BASIC INFO
# =========================
print("\nFirst 5 rows:\n", df.head())

print("\nOverall Statistics:\n")
print(df.describe())

# =========================
# CORRELATION HEATMAP  corr means calculate corelational features
# =========================
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================
# GROUPBY ANALYSIS
# =========================

# Mean age by gender
print("\nMean age grouped by sex:\n")
print(df.groupby("sex")["age"].mean())

# Median age by marital status
print("\nMedian age grouped by marital status:\n")
print(df.groupby("marital_status")["age"].median())

# Standard deviation by sex and marital status
print("\nStandard deviation of age grouped by sex & marital status:\n")
print(df.groupby(["sex", "marital_status"])["age"].std())

# Mean age by income
print("\nMean age grouped by income:\n")
print(df.groupby("income")["age"].mean())

# Mean age by income & sex
print("\nMean age grouped by income and sex:\n")
print(df.groupby(["income", "sex"])["age"].mean())

# Count of marital status
print("\nCount of marital status:\n")
print(df["marital_status"].value_counts())

# Min & Max age by sex
print("\nMin age by sex:\n")
print(df.groupby("sex")["age"].min())

print("\nMax age by sex:\n")
print(df.groupby("sex")["age"].max())

# =========================
# CLEAN VERSION (DROP NA)
# =========================
adult_df = pd.read_csv("adults.csv", skipinitialspace=True)

adult_df.columns = [
    "age","workclass","fnlwgt","education","education_num",
    "marital_status","occupation","relationship","race","sex",
    "capital_gain","capital_loss","hours_per_week","native_country","income"
]

adult_df.replace('?', pd.NA, inplace=True)
adult_df['age'] = pd.to_numeric(adult_df['age'], errors='coerce')

adult_df.dropna(inplace=True)

print("\n=== Clean Adult Dataset Summary ===\n")
print(adult_df.describe())

# =========================
# IRIS DATASET ANALYSIS
# =========================
iris_df = pd.read_csv("iris.csv")

print("\nIris Dataset Preview:\n")
print(iris_df.head())

print("\nUnique Species:\n", iris_df['Species'].unique())

# Group by species
grouped = iris_df.groupby('Species')

print("\nSummary Statistics for Each Species:\n")
print(grouped.describe())

# Detailed statistics per species
for species in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
    if species not in grouped.groups:
        print(f"\nSpecies {species} not found!")
        continue

    sp = grouped.get_group(species)

    print(f"\n=== {species} ===")
    print("Mean:\n", sp.mean(numeric_only=True))
    print("Standard Deviation:\n", sp.std(numeric_only=True))
    print("25th Percentile:\n", sp.quantile(0.25, numeric_only=True))
    print("Median (50th Percentile):\n", sp.quantile(0.5, numeric_only=True))
    print("75th Percentile:\n", sp.quantile(0.75, numeric_only=True))

# Overall percentiles
print("\nOverall Percentiles:\n")
print(iris_df.quantile([0.25, 0.5, 0.75], numeric_only=True))