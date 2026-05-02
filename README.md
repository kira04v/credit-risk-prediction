run - 
source venv/bin/activate
python -m uvicorn app.api:app --reload
streamlit run app/app.py




# Credit Risk Prediction System (End-to-End Machine Learning Project)

An end-to-end machine learning system for assessing loan default risk, combining data preprocessing, predictive modeling, explainability, and real-time deployment.

---

## Overview

Financial institutions must evaluate borrower risk before approving loans. This project simulates a real-world credit risk assessment pipeline that:

* Predicts the probability of loan default
* Classifies applicants into risk categories
* Provides interpretable insights into model decisions
* Exposes predictions through an API and interactive dashboard

---

## Key Features

* Data preprocessing pipeline with cleaning, outlier handling, and feature engineering
* Machine learning model using Random Forest (optionally XGBoost)
* Explainable AI using SHAP to interpret predictions
* Cost-sensitive evaluation to reflect real-world financial impact
* REST API built with FastAPI for real-time inference
* Interactive dashboard built with Streamlit

---

## Architecture

User Input → Streamlit UI → FastAPI → ML Model → Prediction → Risk Decision

---

## Project Structure

```bash
credit-risk/
├── app/
│   ├── api.py            # FastAPI backend
│   └── app.py            # Streamlit UI
│
├── src/
│   ├── preprocess.py     # Data cleaning & feature engineering
│   ├── train.py          # Model training pipeline
│   ├── predict.py        # Inference pipeline
│
├── models/               # Saved model, scaler, and feature columns
├── notebooks/            # EDA and experimentation
├── tests/                # Unit tests
├── requirements.txt
└── README.md
```

---

## Tech Stack

* Language: Python
* Libraries: Pandas, NumPy, Scikit-learn, SHAP
* Backend: FastAPI
* Frontend: Streamlit
* Model persistence: Joblib

---

## Model Performance

* ROC-AUC Score: 0.8701
* Evaluated using precision, recall, confusion matrix
* Decision threshold tuned based on business cost

---

## Business Impact

Traditional evaluation metrics like accuracy are insufficient in financial settings. This project incorporates cost-sensitive analysis:

* False Negatives (approving defaulters) lead to high financial loss
* False Positives (rejecting good customers) lead to opportunity loss

The system enables more informed and financially aligned decision-making.

---

## Example Prediction

Input:

```json
{
  "income": 50000,
  "loan_amount": 200000,
  "property_value": 300000,
  "Credit_Score": 700
}
```

Output:

```json
{
  "default_probability": 0.78,
  "decision": "High Risk (Reject)"
}
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd credit-risk
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train.py
```

### 5. Run the API

```bash
python -m uvicorn app.api:app --reload
```

### 6. Run the UI

```bash
streamlit run app/app.py
```

---

## Testing

```bash
pytest tests/
```

---

## Future Improvements

* Cloud deployment (Render, AWS, or GCP)
* Integration of SHAP explanations into UI
* Real-time data ingestion
* Authentication and access control

---

## Key Learnings

* Designing modular and reusable ML pipelines
* Handling feature consistency between training and inference
* Deploying ML models using APIs and interactive dashboards
* Aligning model evaluation with business objectives

---

## Contact

Ayush Srivastava
Email: (add your email)
LinkedIn/GitHub: (add links)

---

If this project is useful, consider starring the repository.
