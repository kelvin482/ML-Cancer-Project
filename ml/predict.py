# predict.py

import pickle
import numpy as np

from mapping import build_feature_vector

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_risk(user_answers):
    # Build raw feature vector
    vector = build_feature_vector(user_answers)

    # Convert to 2D array
    X = np.array(vector).reshape(1, -1)

    # Scale using training scaler
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]

    # Convert to friendly category
    if prediction == 0:
        return "Higher Pattern Concern"
    else:
        return "Lower Pattern Concern"


if __name__ == "__main__":
    sample_answers = {
        "size": 5,
        "texture": 3,
        "shape": 2,
        "edge": 4,
        "complexity": 3
    }

    result = predict_risk(sample_answers)
    print("Prediction result:", result)
