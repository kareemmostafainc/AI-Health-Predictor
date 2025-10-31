# ==========================================================
# AI Health Predictor - Main Application
# Author: Kareem Mostafa
# ==========================================================

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

from utils.explainability import generate_shap_plot
from data.preprocessing import preprocess_input_data
from model.predict import predict_risk

# ----------------------------------------------------------
# Flask App Configuration
# ----------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = "model/model.pkl"

# Load pre-trained model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------

@app.route("/")
def index():
    """Render home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests from form or API."""
    try:
        # --- Handle Form Data ---
        if request.form:
            age = float(request.form.get("age", 0))
            bmi = float(request.form.get("bmi", 0))
            glucose = float(request.form.get("glucose", 0))
            blood_pressure = float(request.form.get("blood_pressure", 0))
            gender = request.form.get("gender", "male")

            # Prepare data
            input_data = pd.DataFrame([{
                "age": age,
                "bmi": bmi,
                "glucose": glucose,
                "blood_pressure": blood_pressure,
                "gender": gender
            }])

            processed_data = preprocess_input_data(input_data)

            # Predict
            prediction, probability = predict_risk(model, processed_data)

            # SHAP explainability
            shap_plot_path = generate_shap_plot(model, processed_data)

            result = {
                "risk": "High" if prediction == 1 else "Low",
                "probability": round(float(probability) * 100, 2),
                "shap_plot": shap_plot_path
            }

            return render_template("result.html", result=result)

        # --- Handle JSON Request (API Call) ---
        elif request.is_json:
            content = request.get_json()
            df = pd.DataFrame([content])
            processed_data = preprocess_input_data(df)
            prediction, probability = predict_risk(model, processed_data)

            response = {
                "prediction": int(prediction),
                "probability": round(float(probability) * 100, 2)
            }
            return jsonify(response)

        else:
            return jsonify({"error": "Invalid input format"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch", methods=["POST"])
def batch_predict():
    """Batch prediction for uploaded CSV files."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)
        processed_data = preprocess_input_data(df)
        predictions = []
        probabilities = []

        for _, row in processed_data.iterrows():
            pred, prob = predict_risk(model, pd.DataFrame([row]))
            predictions.append(int(pred))
            probabilities.append(round(float(prob) * 100, 2))

        df["Prediction"] = predictions
        df["Risk (%)"] = probabilities

        output_path = "data/batch_results.csv"
        df.to_csv(output_path, index=False)

        return jsonify({"message": "Batch prediction complete", "output_file": output_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------
# Run Application
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
