# model_audit.py

import pickle
import json
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------
# Load saved artifacts
# -----------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_order.json", "r") as f:
    feature_order = json.load(f)

print("Model and artifacts loaded successfully.\n")

# -----------------------------------
# Load original dataset again
# -----------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset loaded.")
print("Dataset shape:", X.shape)
print("Number of features:", len(X.columns))
print("Saved feature count:", len(feature_order))
print()

# -----------------------------------
# 1️⃣ Confirm feature match
# -----------------------------------
if list(X.columns) == feature_order:
    print("✅ Feature order matches training dataset.\n")
else:
    print("❌ Feature order mismatch detected!\n")

# -----------------------------------
# 2️⃣ Confirm scaler alignment
# -----------------------------------
print("Scaler expects feature count:", scaler.mean_.shape[0])

if scaler.mean_.shape[0] == len(feature_order):
    print("✅ Scaler feature count matches.\n")
else:
    print("❌ Scaler feature mismatch!\n")

# -----------------------------------
# 3️⃣ Scale full dataset
# -----------------------------------
X_scaled = scaler.transform(X)

# -----------------------------------
# 4️⃣ Predict on full dataset
# -----------------------------------
y_pred = model.predict(X_scaled)

# -----------------------------------
# 5️⃣ Evaluate performance
# -----------------------------------
accuracy = accuracy_score(y, y_pred)
print("Model Accuracy on full dataset:", accuracy)
print()

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print()

print("Classification Report:")
print(classification_report(y, y_pred))
print()

# -----------------------------------
# 6️⃣ Test multiple real rows manually
# -----------------------------------
print("Testing first 5 rows individually:\n")

for i in range(5):
    row = X.iloc[[i]]
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    actual = y.iloc[i]

    print(f"Row {i+1} -> Actual: {actual}, Predicted: {pred}")

print("\nAudit complete.")
