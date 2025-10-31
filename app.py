# ==========================================================
# AI Health Predictor - Main Application
# Author: Kareem Mostafa
# ==========================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import traceback

# Import compatible modules
from model.predict import load_model, preprocess_input, predict_health_risk
from data.preprocessing import DataPreprocessor
from utils.explainability import ExplainabilityEngine

# ----------------------------------------------------------
# Flask Configuration
# ----------------------------------------------------------
app = Flask(__name__)
MODEL_PATH = "model/model.pkl"

# ----------------------------------------------------------
# Load Model and Scaler
# ----------------------------------------------------------
try:
    model, scaler = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH} -> {e}")
    model, scaler = None, None

# ----------------------------------------------------------
# Home Route
# ----------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------------------------------------------------
# Prediction Route
# ----------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.form:
            # Get form data safely
            data = {
                "age": float(request.form.get("age", 0)),
                "gender": request.form.get("gender", "male"),
                "bmi": float(request.form.get("bmi", 0)),
                "blood_pressure": float(request.form.get("blood_pressure", 0)),
                "glucose_level": float(request.form.get("glucose_level", 0)),
                "cholesterol_level": float(request.form.get("cholesterol_level", 0)),
                "heart_rate": float(request.form.get("heart_rate", 0)),
            }
        elif request.is_json:
            data = request.get_json()
        else:
            return jsonify({"error": "Invalid input format"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Preprocessing
        dp = DataPreprocessor()
        X_scaled, _ = dp.transform(df)

        if model is None or scaler is None:
            return jsonify({"error": "Model not available"}), 503

        processed = preprocess_input(df, scaler)

        # Prediction
        probabilities = predict_health_risk(model, processed)
        probability = float(probabilities[0])
        risk_category = "High" if probability >= 0.5 else "Low"

        # Explainability
        explanation = {}
        try:
            engine = ExplainabilityEngine(MODEL_PATH)
            background = None
            sample_path = os.path.join("data", "sample_data.csv")
            if os.path.exists(sample_path):
                bg_df = pd.read_csv(sample_path).drop(columns=["disease_risk"], errors="ignore")
                background = bg_df.values
            explainer = engine.initialize_explainer(background if background is not None else processed)
            shap_values = engine.compute_shap_values(processed)

            # Summarize SHAP values
            import numpy as np
            arr = shap_values.values if hasattr(shap_values, "values") else shap_values
            mean_imp = np.abs(arr).mean(axis=0).tolist()
            feature_names = list(df.columns)
            explanation = dict(zip(feature_names, [float(x) for x in mean_imp]))
        except Exception:
            explanation = {}

        # Render result page
        return render_template(
            "result.html",
            probability=probability,
            risk_category=risk_category,
            explanation=explanation
        )

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------
# Batch Prediction Route
# ----------------------------------------------------------
@app.route("/batch", methods=["POST"])
def batch_predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)

        if model is None or scaler is None:
            return jsonify({"error": "Model not available"}), 503

        processed = preprocess_input(df, scaler)
        probabilities = predict_health_risk(model, processed)
        df["RiskProbability"] = probabilities

        output_path = "data/batch_results.csv"
        df.to_csv(output_path, index=False)

        return jsonify({
            "message": "Batch prediction completed successfully",
            "output_file": output_path
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------
# Run Application
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
