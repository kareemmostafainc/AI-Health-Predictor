# AI Health Predictor

## Overview
**AI Health Predictor** is a professional machine learning web application designed to estimate the risk of chronic diseases such as **diabetes** and **heart disease** using simple medical data inputs.  
It integrates **predictive modeling**, **explainable AI**, and **interactive visualization** to provide users with transparent and data-driven health insights.

---

## Features
1. **Smart Data Input Form** – Collects user medical attributes such as age, BMI, blood pressure, and glucose level.  
2. **AI Risk Prediction** – Uses a trained Scikit-learn model to estimate the probability of developing a chronic disease.  
3. **Explainable AI (XAI)** – Employs SHAP visualizations to show how each feature influenced the prediction.  
4. **Interactive Dashboard** – Displays prediction results, confidence scores, and personalized health insights.  
5. **Batch Prediction Mode** – Allows uploading CSV files for group health risk prediction.  
6. **Secure and Lightweight** – Built with Flask, ensuring fast, secure, and efficient performance.

---

## Project Structure
AI-Health-Predictor/  
│  
├── README.md                 → Project documentation  
├── requirements.txt          → Python dependencies  
├── .gitignore                → Files and folders excluded from Git  
├── LICENSE                   → MIT open-source license  
├── app.py                    → Flask web application entry point  
│  
├── model/  
│   ├── train_model.py        → Script for training and saving ML model  
│   ├── predict.py            → Handles predictions for web requests  
│   └── model.pkl             → Serialized trained model (generated after training)  
│  
├── data/  
│   ├── sample_data.csv       → Example medical dataset  
│   └── preprocessing.py      → Preprocessing pipeline for cleaning and scaling data  
│  
├── utils/  
│   ├── explainability.py     → SHAP explainability module  
│   └── helpers.py            → Helper functions for model and app  
│  
├── static/  
│   └── assets/  
│       ├── style.css         → Frontend styling  
│       └── script.js         → Client-side interactivity  
│  
└── templates/  
    ├── index.html            → Input form and landing page  
    └── result.html           → Results and visualization dashboard  

---

## Technical Stack
- **Language:** Python 3.11  
- **Framework:** Flask  
- **Machine Learning:** Scikit-learn  
- **Explainable AI:** SHAP  
- **Visualization:** Matplotlib, JavaScript  
- **Frontend:** HTML5, CSS3, JS (Vanilla)  

---

## Installation
1. Clone the repository:  
   `git clone https://github.com/kareemmostafainc/AI-Health-Predictor.git`  
   `cd AI-Health-Predictor`  

2. Install the required dependencies:  
   `pip install -r requirements.txt`  

3. Run the application:  
   `python app.py`  

---

## How It Works
1. The user enters health metrics manually or uploads a CSV file.  
2. The backend model preprocesses the input data using `data/preprocessing.py`.  
3. The trained model in `model/model.pkl` makes predictions using logistic regression.  
4. The output is visualized with an explainable SHAP summary and displayed on `result.html`.  

---

## Example Input/Output

**Example Input (User form):**  
| Age | Gender | BMI | Glucose | Blood Pressure |  
|-----|---------|-----|----------|----------------|  
| 45  | Male    | 28  | 110      | 130            |  

**Example Output:**  
- Disease Risk Probability: **0.72 (High)**  
- Recommendation: “Consider consulting a physician for regular screening.”  
- Explanation: Glucose level and BMI contributed most to the predicted risk.

---

## Model Description
The prediction model is trained on anonymized health datasets and saved as `model/model.pkl`.  
It uses logistic regression to calculate disease probability, while explainability is handled by SHAP visualizations that describe the effect of each medical feature.

---

## Ethical Statement
This project is built for educational and research purposes only.  
It is not a diagnostic tool and should not replace professional medical consultation.  
All data used is synthetic or anonymized to ensure privacy and compliance with ethical standards.

---

## License
This project is licensed under the MIT License – see the LICENSE file for details.

---

## Author
**Kareem Mostafa**  
Email: kareemmostafainc@gmail.com  
GitHub: [github.com/kareemmostafainc](https://github.com/kareemmostafainc)

---

## Acknowledgements
Developed as part of a personal academic initiative to demonstrate applied AI, predictive modeling, and explainable artificial intelligence for university admissions review.
