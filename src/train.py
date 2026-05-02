import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from preprocess import load_data, preprocess


# -------------------------------
# TRAIN FUNCTION
# -------------------------------
def train():

    print("Loading data...")
    df = load_data("data/Loan_Default.csv")

    print("Preprocessing...")
    df = preprocess(df)

    # Separate features and target
    X = df.drop("Status", axis=1)
    y = df["Status"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------------
    # SAVE COLUMN STRUCTURE (IMPORTANT)
    # -------------------------------
    columns = X_train.columns
    joblib.dump(columns, "models/columns.pkl")

    # -------------------------------
    # SCALING
    # -------------------------------
    print("Scaling data...")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "models/scaler.pkl")

    # -------------------------------
    # MODEL TRAINING
    # -------------------------------
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    # -------------------------------
    # EVALUATION
    # -------------------------------
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC Score:", auc)

    # -------------------------------
    # SAVE MODEL
    # -------------------------------
    joblib.dump(model, "models/model.pkl")

    print("Model saved successfully!")


# -------------------------------
# RUN TRAINING
# -------------------------------
if __name__ == "__main__":
    train()