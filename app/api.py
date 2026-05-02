from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict

app = FastAPI(title="Credit Risk API")


# -------------------------------
# INPUT SCHEMA (VERY IMPORTANT)
# -------------------------------
class LoanData(BaseModel):
    income: float
    loan_amount: float
    property_value: float
    Credit_Score: float


# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/")
def home():
    return {"message": "Credit Risk API is running "}


# -------------------------------
# PREDICTION ENDPOINT
# -------------------------------
@app.post("/predict")
def get_prediction(data: LoanData):

    result = predict(data.dict())

    return {
        "default_probability": result["probability"],
        "decision": result["risk_level"]
    }