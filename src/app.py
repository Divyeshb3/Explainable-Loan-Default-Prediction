import os
import json

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

from groq_explain import explain_with_llm

st.set_page_config(
    page_title="Loan Default Predictor",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_model.json")

    encoders = joblib.load("models/encoders.pkl")

    with open("models/model_metadata.json") as f:
        metadata = json.load(f)

    explainer = shap.TreeExplainer(model)

    return model, encoders, metadata, explainer


model, encoders, metadata, explainer = load_artifacts()

feature_names = metadata["feature_names"]
threshold = metadata["best_threshold"]

st.title("🏦 Loan Default Predictor")

st.caption(
    "XGBoost + SHAP explainability + LLM-narrated explanations (Groq / LLaMA 3.3 70B)"
)

# ---------- Sidebar input form ----------
st.sidebar.header("Applicant Details")

def cat_input(label, col):
    options = list(encoders[col].classes_)
    return st.sidebar.selectbox(label, options)

age = st.sidebar.slider("Age", 18, 75, 35)
income = st.sidebar.number_input("Annual Income", 10000, 200000, 60000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 250000, 20000, step=1000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
months_employed = st.sidebar.slider("Months Employed", 0, 480, 60)
num_credit_lines = st.sidebar.slider("Number of Credit Lines", 0, 20, 3)
interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 30.0, 12.0, step=0.1)
loan_term = st.sidebar.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
dti_ratio = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, step=0.01)

education = cat_input("Education", "Education")
employment_type = cat_input("Employment Type", "EmploymentType")
marital_status = cat_input("Marital Status", "MaritalStatus")
has_mortgage = cat_input("Has Mortgage", "HasMortgage")
has_dependents = cat_input("Has Dependents", "HasDependents")
loan_purpose = cat_input("Loan Purpose", "LoanPurpose")
has_cosigner = cat_input("Has Co-Signer", "HasCoSigner")

audience = st.sidebar.radio(
    "Explanation audience",
    ["customer", "analyst"]
)

predict_btn = st.sidebar.button(
    "Predict",
    type="primary"
)

# ---------- Build input row ----------
def build_input_row():
    raw = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
    }

    row = {}

    for col in feature_names:
        val = raw[col]

        if col in encoders:
            val = encoders[col].transform([val])[0]

        row[col] = val

    return pd.DataFrame([row])[feature_names]
# ---------- Main panel ----------
if predict_btn:
    X_input = build_input_row()

    proba = model.predict_proba(X_input)[0, 1]
    prediction = "DEFAULT" if proba >= threshold else "NO DEFAULT"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric(
            "Predicted Default Probability",
            f"{proba:.1%}"
        )

        st.metric(
            "Prediction",
            prediction,
            delta=f"threshold={threshold:.2f}",
            delta_color="off"
        )

# ---------- SHAP for this single row ----------
    shap_values = explainer.shap_values(X_input)
    base_value = explainer.expected_value

    contributions = sorted(
        zip(feature_names, shap_values[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:8]

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))

        feats = [c[0] for c in contributions][::-1]
        vals = [c[1] for c in contributions][::-1]

        colors = [
            "#d62728" if v > 0 else "#2ca02c"
            for v in vals
        ]

        ax.barh(feats, vals, color=colors)
        ax.set_xlabel(
            "SHAP impact (→ increases risk | decreases risk ←)"
        )
        ax.axvline(0, color="black", linewidth=0.8)

        st.pyplot(fig)
        
    # ---------- Build explanation dict for LLM ----------
    explanation = {
        "predicted_default_probability": float(proba),
        "base_value": float(base_value),
        "top_contributions": [
            {
                "feature": f,
                "value": (
                    X_input.iloc[0][f]
                    if f not in encoders
                    else encoders[f].inverse_transform(
                        [int(X_input.iloc[0][f])]
                    )[0]
                ),
                "shap_impact": float(v),
            }
            for f, v in contributions[:6]
        ],
    }

    st.subheader(f"💬 Explanation ({audience})")

    if os.environ.get("GROQ_API_KEY"):
        with st.spinner("Generating explanation..."):
            try:
                narration = explain_with_llm(
                    explanation,
                    audience=audience
                )
                st.write(narration)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
    else:
        st.warning(
            "Set GROQ_API_KEY environment variable to enable LLM narration."
        )

        with st.expander("Raw SHAP contribution data"):
            st.json(explanation)

else:
    st.info(
        "Fill in applicant details in the sidebar and click **Predict**."
    )