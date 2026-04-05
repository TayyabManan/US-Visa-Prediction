"""
Quick evaluation script: loads the saved visaModel from artifact/model.pkl,
reproduces the same feature-engineering that data_ingestion.py performs,
splits the data, and prints classification metrics on the test set.
"""

import pickle
from datetime import date

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from visa_approval_prediction.entity.estimator import visaModel

# ── Constants (same as visa_approval_prediction.constants) ──────────────
CURRENT_YEAR = date.today().year
TARGET_COLUMN = "case_status"

# ── 1. Load the saved model ────────────────────────────────────────────
with open("artifact/model.pkl", "rb") as f:
    model: visaModel = pickle.load(f)

print(f"Loaded model: {model}")

# ── 2. Load the raw dataset ────────────────────────────────────────────
df = pd.read_csv("EasyVisa.csv")
print(f"Dataset shape: {df.shape}")

# ── 3. Feature engineering (mirrors data_ingestion.py) ──────────────────
df["company_age"] = CURRENT_YEAR - df["yr_of_estab"]
df.drop(columns=["case_id", "yr_of_estab"], inplace=True)

# Encode target: Certified = 0, Denied = 1
df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Certified": 0, "Denied": 1})

# ── 4. Stratified train/test split ─────────────────────────────────────
_, test_set = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[TARGET_COLUMN],
)

# ── 5. Separate features and target from the test split ─────────────────
X_test = test_set.drop(columns=[TARGET_COLUMN])
y_test = test_set[TARGET_COLUMN]

# ── 6. Generate predictions ─────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── 7. Print all metrics ────────────────────────────────────────────────
print("\n===== Test-Set Metrics =====")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1       : {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Certified", "Denied"]))
