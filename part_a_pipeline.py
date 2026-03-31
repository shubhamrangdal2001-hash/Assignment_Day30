# Part A — End-to-End ML Pipeline: Logistic Regression on SUV Purchase Dataset
# Dataset: https://www.kaggle.com/datasets/bittupanchal/logistics-regression-on-suv-dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------
# 1. Data Loading & Exploration
# -------------------------------------------------------

df = pd.read_csv("Social_Network_Ads.csv")

print("=== First 5 Rows ===")
print(df.head())

print("\n=== Shape ===")
print(df.shape)

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values ===")
print(df.isnull().sum())

# -------------------------------------------------------
# 2. Data Preprocessing
# -------------------------------------------------------

# No missing values in this dataset, but handling just in case
df.dropna(inplace=True)

# Encode Gender column (Male -> 1, Female -> 0)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Select relevant features
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

print("\n=== Features (X) - First 5 rows ===")
print(X.head())

print("\n=== Target (y) - First 5 values ===")
print(y.head())

# -------------------------------------------------------
# 3. Train-Test Split (80/20)
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n=== Train-Test Split ===")
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

# -------------------------------------------------------
# 4. Feature Scaling (Standard Scaler)
# -------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n=== Scaled Training Data (first 5 rows) ===")
print(X_train_scaled[:5])

# -------------------------------------------------------
# 5. Model Training — Logistic Regression
# -------------------------------------------------------

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

print("\n=== Model Coefficients ===")
print("Intercept :", model.intercept_)
print("Coefficients:", model.coef_)

print("\n[Part A Complete] Model is trained and ready for evaluation.")
