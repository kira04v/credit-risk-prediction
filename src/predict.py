import joblib
import pandas as pd

from src.preprocess import preprocess

# -------------------------------
# LOAD ARTIFACTS (once)
# -------------------------------
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")


# -------------------------------
# ALIGN INPUT DATA TO TRAINING COLUMNS
# -------------------------------
def align_columns(df):

    # Add missing columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Remove extra columns
    df = df[columns]

    return df


# -------------------------------
# MAIN PREDICTION FUNCTION
# -------------------------------
def predict(input_dict):

    # Convert input to DataFrame
    df = pd.DataFrame([input_dict])

    # Apply preprocessing
    df = preprocess(df)

    # Align columns
    df = align_columns(df)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": get_risk_label(probability)
    }


# -------------------------------
# BUSINESS LOGIC (VERY IMPORTANT)
# -------------------------------
def get_risk_label(prob):

    if prob > 0.7:
        return "High Risk (Reject)"
    elif prob > 0.4:
        return "Medium Risk (Review)"
    else:
        return "Low Risk (Approve)"