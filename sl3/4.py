# ==============================
# 1. Import Libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ==============================
# 2. Load Dataset
# ==============================
housing_data = fetch_california_housing()

housing_df = pd.DataFrame(
    housing_data.data,
    columns=housing_data.feature_names
)

# Add target column
housing_df['PRICE'] = housing_data.target


# ==============================
# 3. Basic Info
# ==============================
print("Shape:", housing_df.shape)
print("\nColumns:\n", housing_df.columns)
print("\nSummary:\n", housing_df.describe())


# ==============================
# 4. Introduce Missing Values (for demo)
# ==============================
housing_df.loc[housing_df.sample(frac=0.05).index, 'MedInc'] = np.nan

print("\nMissing Values Before:\n", housing_df.isnull().sum())

# Fill missing values
housing_df.fillna(housing_df.mean(), inplace=True)

print("\nMissing Values After:\n", housing_df.isnull().sum())


# ==============================
# 5. Correlation Heatmap
# ==============================
plt.figure(figsize=(10, 8))
sns.heatmap(housing_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# ==============================
# 6. Outlier Detection (IQR)
# ==============================
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    return outliers, lower_bound, upper_bound


# Detect outliers in PRICE
outliers, lower_bound, upper_bound = detect_outliers_iqr(housing_df, 'PRICE')

print("\nNumber of outliers:", len(outliers))
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


# ==============================
# 7. Boxplot for Outliers
# ==============================
plt.figure(figsize=(6, 5))
housing_df.boxplot(column='PRICE')
plt.title("Boxplot of PRICE")
plt.show()


# ==============================
# 8. Remove Outliers
# ==============================
housing_df_clean = housing_df[
    (housing_df['PRICE'] >= lower_bound) &
    (housing_df['PRICE'] <= upper_bound)
]

print("\nOriginal Shape:", housing_df.shape)
print("Cleaned Shape:", housing_df_clean.shape)
print("Removed Outliers:", len(housing_df) - len(housing_df_clean))


# ==============================
# 9. Feature & Target Split
# ==============================
X = housing_df.drop("PRICE", axis=1)
y = housing_df["PRICE"]

X_clean = housing_df_clean.drop("PRICE", axis=1)
y_clean = housing_df_clean["PRICE"]


# ==============================
# 10. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=10
)


# ==============================
# 11. Feature Scaling
# ==============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train) #for training data
X_test_scaled = scaler.transform(X_test)    #for testing data

X_train_clean_scaled = scaler.fit_transform(X_train_clean)
X_test_clean_scaled = scaler.transform(X_test_clean)


# ==============================
# 12. Train Model
# ==============================
#fit() learns all weights and biases
model = LinearRegression()
model.fit(X_train_scaled, y_train)

model_clean = LinearRegression()
model_clean.fit(X_train_clean_scaled, y_train_clean)


# ==============================
# 13. Predictions
# ==============================
#"The model predicts target values for unseen test data using the learned regression equation."
y_pred = model.predict(X_test_scaled)
y_pred_clean = model_clean.predict(X_test_clean_scaled)


# ==============================
# 14. Evaluation
# ==============================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

rmse_clean = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
r2_clean = r2_score(y_test_clean, y_pred_clean)

print("\n==============================")
print("Original Model Performance")
print("==============================")
print("RMSE:", rmse)
#r2 tells us if it is perfect prediction(1) ,No learning (0),Worse than guessing(less than 0)
print("R2 Score:", round(r2, 2))

print("\n==============================")
print("Cleaned Model Performance")
print("==============================")
print("RMSE:", rmse_clean)
print("R2 Score:", round(r2_clean, 2))


# ==============================
# 15. Comparison
# ==============================
print("\n==============================")
print("Comparison")
print("==============================")
print("Original RMSE:", rmse, " | Cleaned RMSE:", rmse_clean)
print("Original R2:", round(r2, 2), " | Cleaned R2:", round(r2_clean, 2))


# ==============================
# 16. Actual vs Predicted Plot
# ==============================
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
#The scatter plot compares actual vs predicted values, and closeness to the diagonal line indicates model accuracy
# Perfect prediction line
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    'r--'
)

plt.show()