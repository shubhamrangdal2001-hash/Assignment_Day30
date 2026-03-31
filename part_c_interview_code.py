# Part C — Interview-Ready Code Snippet
# Q2: Write code to perform train-test split and feature scaling

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Features and target
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training size : {X_train.shape[0]}")
print(f"Testing size  : {X_test.shape[0]}")

# --- Standard Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled  = scaler.transform(X_test)         # only transform on test

print("\nFirst 3 rows — before scaling:")
print(X_train.values[:3])

print("\nFirst 3 rows — after scaling:")
print(X_train_scaled[:3])
