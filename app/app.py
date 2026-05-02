import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("Credit Risk Prediction System")
st.markdown("Enter applicant details to assess loan default risk")

# -------------------------------
# INPUT FORM
# -------------------------------
income = st.number_input("Income", min_value=0.0, value=50000.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0)
property_value = st.number_input("Property Value", min_value=0.0, value=300000.0)
credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=700.0)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict Risk"):

    payload = {
        "income": income,
        "loan_amount": loan_amount,
        "property_value": property_value,
        "Credit_Score": credit_score
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        result = response.json()

        st.subheader("Prediction Result")

        st.metric("Default Probability", f"{result['default_probability']:.2f}")

        decision = result["decision"]

        if "High" in decision:
            st.error(f" {decision}")
        elif "Medium" in decision:
            st.warning(f" {decision}")
        else:
            st.success(f" {decision}")

    except Exception as e:
        st.error("API not running. Please start FastAPI server.")