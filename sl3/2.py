# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# %%
import numpy as np
import pandas as pd

np.random.seed(42)

genders = ["male", "female"]
race_groups = ["group A", "group B", "group C", "group D", "group E"]

parent_education = [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
]

lunch_types = ["standard", "free/reduced"]
test_prep = ["Incompleted", "completed"]

n = 20

data = {
    "gender": np.random.choice(genders, n),
    "race/ethnicity": np.random.choice(race_groups, n),
    "parental level of education": np.random.choice(parent_education, n),
    "lunch": np.random.choice(lunch_types, n),
    "test preparation course": np.random.choice(test_prep, n),
    "math score": np.random.randint(40, 100, n).astype(float),
    "reading score": np.random.randint(40, 100, n).astype(float),
    "writing score": np.random.randint(40, 100, n).astype(float)
}

df = pd.DataFrame(data)
print(df.head())

# %%
numeric_cols = ["math score", "reading score", "writing score"]

df.loc[0, 'math score'] = 10
df.loc[1, 'math score'] = 5
df.loc[2, 'math score'] = 0
df.loc[3, 'math score'] = 160
df.loc[4, 'math score'] = 180

for col in numeric_cols:
    df.loc[df.sample(frac=0.1).index, col] = np.nan

df

# %%
print("Dataset Info:")
print(df.describe())

print("\nShape:", df.shape)

print("\nMissing Values:\n", df.isnull().sum())

# %%
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

print("After Filling Missing Values:\n", df.isnull().sum())

print(df.head())

# %%
plt.figure()
plt.boxplot(df['math score'])
plt.title("Boxplot of Math Score")
plt.xlabel("Math Score")
plt.show()

# %%
Q1 = df['math score'].quantile(0.25)
Q3 = df['math score'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR    #18.75
upper_limit = Q3 + 1.5 * IQR    #138.75

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

outliers = df[(df['math score'] < lower_limit) | (df['math score'] > upper_limit)]
outliers

# %%
df_clean = df[(df['math score'] >= lower_limit) & (df['math score'] <= upper_limit)]

print("Shape after removing outliers:", df_clean.shape)

# %%
#Min-Max scaling normalizes the data between 0 and 1 using min and max values, improving model performance.
scaler = MinMaxScaler()
df_clean['math_scaled'] = scaler.fit_transform(df_clean[['math score']])

# Z-score normalization  (x-mean / standard deviation)
df_clean['math_zscore'] = (
    df_clean['math score'] - df_clean['math score'].mean()
) / df_clean['math score'].std()

# Log transformation used when data is skewed
df_clean['math_log'] = np.log(df_clean['math score'])

print(df_clean.head())

# %%
plt.figure()
plt.hist(df_clean['math score'], bins=10)
plt.title("Histogram of Math Score")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.show()

# %%
print("Final Statistical Summary:\n")
print(df_clean.describe())


