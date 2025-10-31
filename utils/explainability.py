# ==========================================================
# AI Health Predictor - Explainability Module
# Author: Kareem Mostafa
# ==========================================================

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ----------------------------------------------------------
# SHAP Explainability Handler
# ----------------------------------------------------------
class ExplainabilityEngine:
    """Generates SHAP-based explanations for AI Health Predictor model."""

    def __init__(self, model_path: str = "model/model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        saved = joblib.load(model_path)
        self.model = saved["model"]
        self.scaler = saved["scaler"]
        self.explainer = None

    # ------------------------------------------------------
    # Initialize SHAP Explainer
    # ------------------------------------------------------
    def initialize_explainer(self, background_data: np.ndarray):
        """Create SHAP explainer using model and background data."""
        self.explainer = shap.Explainer(self.model, background_data)
        return self.explainer

    # ------------------------------------------------------
    # Generate SHAP Values
    # ------------------------------------------------------
    def compute_shap_values(self, X: np.ndarray):
        """Compute SHAP values for input features."""
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call initialize_explainer() first.")
        shap_values = self.explainer(X)
        return shap_values

    # ------------------------------------------------------
    # Plot Single Prediction Explanation
    # ------------------------------------------------------
    def plot_individual_explanation(self, shap_values, feature_names, instance_index=0):
        """Visualize feature impact for a single prediction."""
        plt.figure(figsize=(8, 5))
        shap.plots.bar(shap_values[instance_index], show=False)
        plt.title("Feature Impact on Prediction", fontsize=12)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------
    # Plot Global Feature Importance
    # ------------------------------------------------------
    def plot_global_importance(self, shap_values, feature_names):
        """Visualize overall feature importance."""
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------
# Example Usage (for testing)
# ----------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    # Load sample data
    df = pd.read_csv("data/sample_data.csv")
    X = df.drop(columns=["disease_risk"], errors="ignore").values

    engine = ExplainabilityEngine()
    explainer = engine.initialize_explainer(X)
    shap_values = engine.compute_shap_values(X)

    print("Explainability module executed successfully.")
    print(f"SHAP values shape: {np.array(shap_values.values).shape}")
