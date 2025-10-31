# AI Health Predictor

## Overview
AI Health Predictor is a professional machine learning web application designed to estimate the risk of chronic diseases such as diabetes and heart disease using simple medical data inputs.  
It combines predictive analytics, explainable AI, and interactive visualization to help users understand their health profile in a data-driven way.

---

## Features
1. **Data Input Form** – Users can enter medical attributes such as age, BMI, blood pressure, and glucose level.  
2. **AI Model Prediction** – A trained ML model (Scikit-learn) generates a probability score for disease risk.  
3. **Explainable AI** – SHAP-based visual explanations clarify how each feature contributed to the model’s decision.  
4. **Result Dashboard** – Displays prediction results with dynamic charts and personalized health insights.  
5. **Batch CSV Upload** – Allows uploading multiple records for group health prediction.

---

## Project Structure
AI-Health-Predictor/  
│  
├── README.md  
├── requirements.txt  
├── .gitignore  
├── LICENSE  
├── app.py  
│  
├── model/  
│   ├── train_model.py  
│   ├── predict.py  
│   └── model.pkl  
│  
├── data/  
│   ├── sample_data.csv  
│   └── preprocessing.py  
│  
├── utils/  
│   ├── explainability.py  
│   └── helpers.py  
│  
├── static/  
│   └── assets/  
│       ├── style.css  
│       └── script.js  
│  
└── templates/  
    ├── index.html  
    └── result.html  

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
