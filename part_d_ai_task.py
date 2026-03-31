# Part D — AI-Augmented Task
# Prompt used: "Explain Logistic Regression with Python example using sklearn on SUV dataset."
#
# AI Output (cleaned, verified, and commented by student):
# Steps verified: Data loading, encoding, split, scaling, training, prediction, accuracy — All present ✓
# Code correctness: Verified — runs without errors ✓

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv("Social_Network_Ads.csv")

# Encode categorical column
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Features & target
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# Train Logistic Regression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Results
print("Accuracy       :", accuracy_score(y_test, y_pred) * 100, "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ----------------------------------------------------------
# Student Evaluation Notes:
# 1. Is the code correct?
#    YES — The pipeline covers all standard ML steps correctly.
#    One minor note: ideally random_state should be consistent
#    across split and model, but both work fine separately.
#
# 2. Are the steps complete?
#    YES — All required steps are present:
#      - Data loading        ✓
#      - Encoding            ✓
#      - Feature selection   ✓
#      - Train-test split    ✓
#      - Feature scaling     ✓
#      - Model training      ✓
#      - Prediction          ✓
#      - Accuracy + CM       ✓
#
# Missing (minor): No visualization of decision boundary,
# but that was listed as optional in the assignment.
# ----------------------------------------------------------
