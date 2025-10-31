# ==========================================================
# AI Health Predictor - Model Training Script
# Author: Kareem Mostafa
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
DATA_PATH = "data/sample_data.csv"
MODEL_PATH = "model/model.pkl"

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

# ----------------------------------------------------------
# Data Preprocessing
# ----------------------------------------------------------
def preprocess_data(df: pd.DataFrame):
    """Clean and prepare dataset for training."""
    df = df.copy()

    # Encode gender if exists
    if "gender" in df.columns:
        encoder = LabelEncoder()
        df["gender"] = encoder.fit_transform(df["gender"])

    # Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))

    # Features and target
    X = df.drop(columns=["disease_risk"], errors="ignore")
    y = df["disease_risk"] if "disease_risk" in df.columns else np.random.randint(0, 2, len(df))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# ----------------------------------------------------------
# Model Training
# ----------------------------------------------------------
def train_model(X, y):
    """Train logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

# ----------------------------------------------------------
# Model Evaluation
# ----------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return acc, report

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------
def save_model(model, scaler, path=MODEL_PATH):
    """Save trained model and scaler to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    print(f"Model saved successfully at: {path}")

# ----------------------------------------------------------
# Main Execution
# ----------------------------------------------------------
def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    X, y, scaler = preprocess_data(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    acc, report = evaluate_model(model, X_test, y_test)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print(f"Classification Report:\n{report}")

    print("Saving model...")
    save_model(model, scaler)

    print("Training complete.")

if __name__ == "__main__":
    main()
