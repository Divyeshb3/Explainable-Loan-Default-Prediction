import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_recall_curve,
)

os.makedirs("models", exist_ok=True)

RANDOM_STATE = 42

# ---------- 1. Load ----------
df = pd.read_csv("data/Loan_default.csv")
df = df.drop(columns=["LoanID"])  # not predictive

# ---------- 2. Encode categoricals ----------
cat_cols = ["Education", "EmploymentType", "MaritalStatus",
            "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

joblib.dump(encoders, "models/encoders.pkl")

# ---------- 3. Train/test split ----------
X = df.drop(columns=["Default"])
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train default rate: {y_train.mean():.3f}, Test default rate: {y_test.mean():.3f}")

# ---------- 4. Handle class imbalance ----------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ---------- 5. Train XGBoost ----------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ---------- 6. Evaluate ----------
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC: {auc:.4f}")
print("\nClassification report (default threshold 0.5):")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# Find a better threshold for recall on the minority class (defaults matter more to catch)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print(f"Best F1 threshold: {best_threshold:.3f} (F1={f1_scores[best_idx]:.3f})")

y_pred_tuned = (y_proba >= best_threshold).astype(int)
print("\nClassification report (tuned threshold):")
print(classification_report(y_test, y_pred_tuned, target_names=["No Default", "Default"]))

# ---------- 7. Save everything ----------
model.save_model("models/xgb_model.json")
X_test.to_csv("models/X_test.csv", index=False)
y_test.to_csv("models/y_test.csv", index=False)

metadata = {
    "auc": float(auc),
    "best_threshold": float(best_threshold),
    "feature_names": list(X.columns),
    "cat_cols": cat_cols,
}
with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("""
Saved artifacts:
- models/xgb_model.json
- models/encoders.pkl
- models/model_metadata.json
- models/X_test.csv
- models/y_test.csv
""")