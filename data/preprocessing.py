# ==========================================================
# AI Health Predictor - Data Preprocessing Module
# Author: Kareem Mostafa
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ----------------------------------------------------------
# Preprocessing Class
# ----------------------------------------------------------
class DataPreprocessor:
    """Handles data cleaning, encoding, and scaling for health prediction."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.fitted = False

    # ------------------------------------------------------
    # Data Cleaning
    # ------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean missing or invalid data."""
        df = df.copy()

        # Drop duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))

        # Remove outliers (basic rule-based filter)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df = df[(df[col] >= lower) & (df[col] <= upper)]

        return df

    # ------------------------------------------------------
    # Feature Encoding
    # ------------------------------------------------------
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features such as gender."""
        df = df.copy()
        if "gender" in df.columns:
            df["gender"] = self.encoder.fit_transform(df["gender"].astype(str))
        return df

    # ------------------------------------------------------
    # Feature Scaling
    # ------------------------------------------------------
    def scale_features(self, df: pd.DataFrame) -> np.ndarray:
        """Scale numerical features using StandardScaler."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaled_data = self.scaler.fit_transform(df[numeric_cols])
        self.fitted = True
        return scaled_data, numeric_cols

    # ------------------------------------------------------
    # Full Preprocessing Pipeline
    # ------------------------------------------------------
    def transform(self, df: pd.DataFrame):
        """Run complete preprocessing pipeline."""
        df = self.clean_data(df)
        df = self.encode_features(df)
        scaled_data, cols = self.scale_features(df)
        return scaled_data, cols


# ----------------------------------------------------------
# Example Usage (for testing)
# ----------------------------------------------------------
if __name__ == "__main__":
    path = "data/sample_data.csv"
    df = pd.read_csv(path)
    pre = DataPreprocessor()
    X_scaled, cols = pre.transform(df)
    print("Data successfully preprocessed.")
    print(f"Scaled Features Shape: {X_scaled.shape}")
    print(f"Feature Columns: {list(cols)}")
