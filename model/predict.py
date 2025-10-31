# ==========================================================
# AI Health Predictor - Model Prediction Script
# Author: Kareem Mostafa
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
MODEL_PATH = "model/model.pkl"

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------
def load_model(path: str):
    """Load the trained model and scaler from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    model_data = joblib.load(path)
    model = model_data.get("model")
    scaler = model_data.get("scaler")
    return model, scaler

# ----------------------------------------------------------
# Preprocess Input Data
# ----------------------------------------------------------
def preprocess_input(input_data: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Preprocess new input data using the saved scaler."""
    required_features = scaler.mean_.shape[0]
    if input_data.shape[1] != required_features:
        raise ValueError(f"Expected {required_features} features, got {input_data.shape[1]}")
    processed = scaler.transform(input_data)
    return processed

# ----------------------------------------------------------
# Generate Prediction
# ----------------------------------------------------------
def predict_health_risk(model, processed_data: np.ndarray) -> np.ndarray:
    """Generate disease risk predictions."""
    predictions = model.predict_proba(processed_data)[:, 1]
    return predictions

# ----------------------------------------------------------
# Predict from CSV
# ----------------------------------------------------------
def predict_from_csv(csv_path: str):
    """Generate predictions for multiple records from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    data = pd.read_csv(csv_path)
    model, scaler = load_model(MODEL_PATH)

    processed = preprocess_input(data, scaler)
    predictions = predict_health_risk(model, processed)

    data["predicted_risk"] = predictions
    output_path = "data/predictions_output.csv"
    data.to_csv(output_path, index=False)

    print(f"Predictions saved successfully to {output_path}")
    return data

# ----------------------------------------------------------
# Predict for Single Input
# ----------------------------------------------------------
def predict_single(input_dict: dict):
    """Generate prediction for a single user input (dictionary format)."""
    model, scaler = load_model(MODEL_PATH)
    df = pd.DataFrame([input_dict])
    processed = preprocess_input(df, scaler)
    prediction = predict_health_risk(model, processed)
    risk_percentage = float(prediction[0]) * 100
    print(f"Predicted Health Risk: {risk_percentage:.2f}%")
    return risk_percentage

# ----------------------------------------------------------
# Main Execution
# ----------------------------------------------------------
def main():
    print("Model Prediction Utility")
    print("----------------------------------------------------------")

    sample_input = {
        "age": 45,
        "gender": 1,
        "bmi": 28.5,
        "blood_pressure": 130,
        "glucose": 95,
        "cholesterol": 180
    }

    print("Running single prediction test...")
    predict_single(sample_input)

    print("\nRunning batch prediction test from CSV (if available)...")
    csv_file = "data/sample_data.csv"
    if os.path.exists(csv_file):
        predict_from_csv(csv_file)
    else:
        print("CSV file not found. Skipping batch prediction test.")

if __name__ == "__main__":
    main()
