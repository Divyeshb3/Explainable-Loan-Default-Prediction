"""
Generate Evaluation Dataset

This script creates a representative evaluation set for
LLM explainability testing.

It:

- Loads the trained XGBoost model
- Loads the test dataset
- Computes SHAP explanations
- Samples applicants from
    • Low-risk
    • Medium-risk
    • High-risk
- Saves 25 representative cases for LLM evaluation.

Output:
outputs/eval_set.json
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
import joblib

model = xgb.XGBClassifier()
model.load_model("models/xgb_model.json")

X_test = pd.read_csv("models/X_test.csv")
y_test = pd.read_csv("models/y_test.csv")["Default"]
encoders = joblib.load("models/encoders.pkl")

explainer = shap.TreeExplainer(model)

proba_all = model.predict_proba(X_test)[:, 1]

# Stratify: low risk (<0.2), medium risk (0.2-0.6), high risk (>0.6)
np.random.seed(42)
low_idx = np.where(proba_all < 0.2)[0]
med_idx = np.where((proba_all >= 0.2) & (proba_all <= 0.6))[0]
high_idx = np.where(proba_all > 0.6)[0]

sample_idx = np.concatenate([
    np.random.choice(low_idx, size=9, replace=False),
    np.random.choice(med_idx, size=8, replace=False),
    np.random.choice(high_idx, size=8, replace=False),
])
np.random.shuffle(sample_idx)

def explain_row(idx):
    row = X_test.iloc[idx]
    row_shap = explainer.shap_values(X_test.iloc[[idx]])[0]
    base_value = explainer.expected_value

    decoded = {}
    for col in X_test.columns:
        if col in encoders:
            decoded[col] = encoders[col].inverse_transform([int(row[col])])[0]
        else:
            decoded[col] = row[col]

    contributions = sorted(
        zip(X_test.columns, row_shap, [decoded[c] for c in X_test.columns]),
        key=lambda x: abs(x[1]), reverse=True
    )

    proba = model.predict_proba(X_test.iloc[[idx]])[0, 1]

    return {
        "row_index": int(idx),
        "predicted_default_probability": float(proba),
        "actual_label": int(y_test.iloc[idx]),
        "base_value": float(base_value),
        "top_contributions": [
            {"feature": f, "value": v, "shap_impact": float(s)}
            for f, s, v in contributions[:6]
        ]
    }

eval_examples = [explain_row(i) for i in sample_idx]

with open("outputs/eval_set.json", "w") as f:
    json.dump(eval_examples, f, indent=2)

risk_bands = [
    "low" if e["predicted_default_probability"] < 0.2
    else "high" if e["predicted_default_probability"] > 0.6
    else "medium"
    for e in eval_examples
]

print("=" * 60)
print("EVALUATION SET CREATED")
print("=" * 60)
print(f"Saved {len(eval_examples)} examples to outputs/eval_set.json")
print(
    f"Risk distribution: low={risk_bands.count('low')}, "
    f"medium={risk_bands.count('medium')}, "
    f"high={risk_bands.count('high')}"
)
print(f"Actual defaults in sample: {sum(e['actual_label'] for e in eval_examples)}")