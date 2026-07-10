import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

os.makedirs("outputs", exist_ok=True)
# ---------- Load model & data ----------
model = xgb.XGBClassifier()
model.load_model("models/xgb_model.json")

X_test = pd.read_csv("models/X_test.csv")
y_test = pd.read_csv("models/y_test.csv")["Default"]
encoders = joblib.load("models/encoders.pkl")
# Load metadata for future use in the Streamlit application.
with open("models/model_metadata.json") as f:
    metadata = json.load(f)

# ---------- SHAP explainer (TreeExplainer is fast & exact for XGBoost) ----------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ---------- Global importance ----------
global_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": global_importance
}).sort_values("mean_abs_shap", ascending=False)

print("Global feature importance (mean |SHAP value|):")
print(importance_df.to_string(index=False))
importance_df.to_csv(
    "outputs/global_importance.csv",
    index=False
)

# ---------- Helper: decode a row back to human-readable + get its SHAP breakdown ----------
def explain_row(idx):
    row = X_test.iloc[idx]
    row_shap = shap_values[idx]
    base_value = explainer.expected_value

    # decode categoricals back to labels
    decoded = {}
    for col in X_test.columns:
        if col in encoders:
            decoded[col] = encoders[col].inverse_transform([int(row[col])])[0]
        else:
            decoded[col] = row[col]

    contributions = sorted(
        zip(X_test.columns, row_shap, [decoded[c] for c in X_test.columns]),
        key=lambda x: abs(x[1]),
        reverse=True
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

# ---------- Save a handful of example explanations (mix of default / no-default) ----------
default_idx = y_test[y_test == 1].index[:3].tolist()
nodefault_idx = y_test[y_test == 0].index[:3].tolist()

examples = [explain_row(X_test.index.get_loc(i)) for i in default_idx + nodefault_idx]

with open("outputs/shap_examples.json","w") as f:
    json.dump(examples, f, indent=2)

print("""
Saved outputs:
- outputs/global_importance.csv
- outputs/shap_examples.json
- outputs/shap_values.npy
- outputs/shap_base_value.json
""")
print("\nExample explanation (row 0):")
print(json.dumps(examples[0], indent=2))

# ---------- Save shap_values array + explainer base value for reuse in the app ----------
np.save("outputs/shap_values.npy", shap_values)
with open("outputs/shap_base_value.json", "w") as f:
    json.dump({"base_value": float(explainer.expected_value)}, f)