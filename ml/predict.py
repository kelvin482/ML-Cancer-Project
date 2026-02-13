# predict.py
# -----------------------------------
# Handles ML prediction using a trained model
# -----------------------------------

import json
import pickle

import numpy as np
import pandas as pd

from mapping import build_feature_vector


# -----------------------------------
# Load feature configuration
# -----------------------------------
with open("feature_order.json", "r") as f:
    FEATURE_ORDER = json.load(f)


# -----------------------------------
# Load trained ML artifacts
# -----------------------------------
with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)


# -----------------------------------
# Prediction function (Django will use this)
# -----------------------------------
def predict_risk(user_answers):
    """
    Takes user answers as a dictionary and returns
    a human-friendly risk category.
    """

    # Convert user answers into numeric feature vector
    vector = build_feature_vector(user_answers)

    # Wrap input in DataFrame to preserve feature names
    X_df = pd.DataFrame([vector], columns=FEATURE_ORDER)

    # Apply same scaling used during training
    X_scaled = SCALER.transform(X_df)

    # Make prediction
    prediction = MODEL.predict(X_scaled)[0]

    # Convert numeric output to readable result
    return (
        "Higher Pattern Concern"
        if prediction == 0
        else "Lower Pattern Concern"
    )


# -----------------------------------
# Simple local tests (safe to remove later)
# -----------------------------------
if __name__ == "__main__":

    print("Test 1: Lower-like values")
    test_case_1 = {
        "size": 2,
        "texture": 1,
        "shape": 1,
        "edge": 2,
        "complexity": 1
    }
    print("Result:", predict_risk(test_case_1))

    print("\nTest 2: Higher-like values")
    test_case_2 = {
        "size": 8,
        "texture": 7,
        "shape": 6,
        "edge": 8,
        "complexity": 7
    }
    print("Result:", predict_risk(test_case_2))
