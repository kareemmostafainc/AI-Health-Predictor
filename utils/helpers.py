# ==========================================================
# AI Health Predictor - Utility Helper Functions
# Author: Kareem Mostafa
# ==========================================================

import os
import json
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------
# File Management Utilities
# ----------------------------------------------------------
def load_model(model_path: str):
    """Load trained model and scaler from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    data = joblib.load(model_path)
    return data["model"], data["scaler"]


def save_json(data: dict, output_path: str):
    """Save dictionary data as a formatted JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"JSON file saved successfully at: {output_path}")


# ----------------------------------------------------------
# Data Validation Utilities
# ----------------------------------------------------------
def validate_input_data(df: pd.DataFrame, expected_columns: list):
    """Ensure the input dataframe contains all required columns."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def sanitize_inputs(data: dict) -> pd.DataFrame:
    """Convert user input dictionary to valid DataFrame."""
    df = pd.DataFrame([data])
    df = df.replace(["", None, "NaN"], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    return df


# ----------------------------------------------------------
# Numeric Utilities
# ----------------------------------------------------------
def normalize_values(values):
    """Normalize list or array of numbers between 0 and 1."""
    arr = np.array(values, dtype=float)
    min_val, max_val = np.min(arr), np.max(arr)
    if min_val == max_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def calculate_health_score(probability: float) -> str:
    """Convert probability into readable health risk category."""
    if probability < 0.3:
        return "Low Risk"
    elif 0.3 <= probability < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"


# ----------------------------------------------------------
# Example Usage (for testing)
# ----------------------------------------------------------
if __name__ == "__main__":
    model, scaler = load_model("model/model.pkl")
    print("Model and scaler loaded successfully.")

    data = {"age": 45, "gender": "Male", "bmi": 28.5, "blood_pressure": 130, "glucose_level": 110,
            "cholesterol_level": 205, "heart_rate": 82}
    df = sanitize_inputs(data)
    print("Sanitized DataFrame:")
    print(df)
