# Part B — Model Evaluation, Decision Boundary & Test Size Comparison
# Dataset: SUV Purchase Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------------------------------
# Helper: build and evaluate a model for a given test_size
# -------------------------------------------------------

def build_and_evaluate(X, y, test_size, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(random_state=random_state)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    return model, scaler, X_train_sc, X_test_sc, y_test, y_pred, acc, cm

# -------------------------------------------------------
# Load & preprocess
# -------------------------------------------------------

df = pd.read_csv("Social_Network_Ads.csv")
df.dropna(inplace=True)

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

X = df[["Age", "EstimatedSalary"]].values
y = df["Purchased"].values

# -------------------------------------------------------
# 1. Model Evaluation (80/20 split)
# -------------------------------------------------------

model, scaler, X_train_sc, X_test_sc, y_test, y_pred, acc, cm = \
    build_and_evaluate(pd.DataFrame(X, columns=["Age", "EstimatedSalary"]),
                       pd.Series(y), test_size=0.20)

print("=== Model Evaluation (80/20 split) ===")
print(f"Accuracy : {acc * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nInterpretation:")
print(f"  True Negatives  (Not Purchased, predicted correctly) : {cm[0][0]}")
print(f"  False Positives (Not Purchased, predicted as Bought) : {cm[0][1]}")
print(f"  False Negatives (Purchased, predicted as Not Bought) : {cm[1][0]}")
print(f"  True Positives  (Purchased, predicted correctly)     : {cm[1][1]}")

# -------------------------------------------------------
# 2. Visualization — Decision Boundary (2D)
# -------------------------------------------------------

def plot_decision_boundary(model, scaler, X_raw, y_raw, title, filename):
    X_sc = scaler.transform(X_raw)

    # Create mesh grid
    x_min, x_max = X_sc[:, 0].min() - 0.5, X_sc[:, 0].max() + 0.5
    y_min, y_max = X_sc[:, 1].min() - 0.5, X_sc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(X_sc[:, 0], X_sc[:, 1], c=y_raw,
                cmap="RdBu", edgecolors="k", s=30)
    not_bought = mpatches.Patch(color="red",  alpha=0.5, label="Not Purchased")
    bought     = mpatches.Patch(color="blue", alpha=0.5, label="Purchased")
    plt.legend(handles=[not_bought, bought])
    plt.xlabel("Age (scaled)")
    plt.ylabel("Estimated Salary (scaled)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Decision boundary saved → {filename}")

# Re-train on full dataset so we can visualize all points
full_scaler = StandardScaler()
X_sc_full   = full_scaler.fit_transform(X)
full_model  = LogisticRegression(random_state=42)
full_model.fit(X_sc_full, y)

plot_decision_boundary(full_model, full_scaler, X, y,
                       "Decision Boundary — Logistic Regression (SUV Dataset)",
                       "decision_boundary.png")

# Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Not Purchased", "Purchased"])
disp.plot(colorbar=False)
plt.title("Confusion Matrix (80/20 split)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()
print("Confusion matrix saved → confusion_matrix.png")

# -------------------------------------------------------
# 3. Improvement — Compare different test sizes
# -------------------------------------------------------

print("\n=== Accuracy for Different Test Sizes ===")
print(f"{'Split':<12} {'Train Samples':<16} {'Test Samples':<14} {'Accuracy'}")
print("-" * 55)

splits = [0.20, 0.25, 0.30]
labels = ["80/20", "75/25", "70/30"]

for label, ts in zip(labels, splits):
    *_, acc_i, _ = build_and_evaluate(
        pd.DataFrame(X, columns=["Age", "EstimatedSalary"]),
        pd.Series(y), test_size=ts
    )
    n_total = len(y)
    n_test  = int(n_total * ts)
    n_train = n_total - n_test
    print(f"{label:<12} {n_train:<16} {n_test:<14} {acc_i * 100:.2f}%")

print("\n[Part B Complete] Evaluation, visualizations, and comparison done.")
