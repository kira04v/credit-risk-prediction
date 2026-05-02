import pandas as pd
import numpy as np

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# -------------------------------
# CLEAN DATA
# -------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Drop unnecessary columns
    df = df.drop(columns=["ID", "year"], errors="ignore")

    leak_cols = [
        "Interest_rate_spread",
        "Upfront_charges",
        "rate_of_interest",
        "dtir1",
        "credit_type"
    ]
    df = df.drop(columns=leak_cols, errors="ignore")

    # Separate column types
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Handle missing values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # Winsorization (outlier handling)
    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    return df


# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df["loan_income_ratio"] = df["loan_amount"] / (df["income"] + 1)
    df["loan_property_ratio"] = df["loan_amount"] / (df["property_value"] + 1)
    df["credit_income_ratio"] = df["Credit_Score"] / (df["income"] + 1)
    df["loan_credit_ratio"] = df["loan_amount"] / (df["Credit_Score"] + 1)
    df["property_income_ratio"] = df["property_value"] / (df["income"] + 1)

    # Remove infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


# -------------------------------
# ENCODING
# -------------------------------
def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


# -------------------------------
# FULL PIPELINE
# -------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_data(df)

    return df
