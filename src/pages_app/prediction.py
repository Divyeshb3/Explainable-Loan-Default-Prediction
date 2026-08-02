import pandas as pd
import streamlit as st
from common import load_artifacts, risk_tier

model, encoders, metadata, explainer = load_artifacts()
feature_names = metadata["feature_names"]
threshold = metadata["best_threshold"]

st.title("📝 Prediction input")
st.caption("Enter the applicant's details, then predict their default risk.")

tab1, tab2, tab3 = st.tabs(["Applicant info", "Loan details", "Financial info"])

with tab1:
    age = st.slider("Age", 18, 75, 35)
    education = st.selectbox("Education", list(encoders["Education"].classes_))
    employment_type = st.selectbox("Employment type", list(encoders["EmploymentType"].classes_))
    marital_status = st.selectbox("Marital status", list(encoders["MaritalStatus"].classes_))
    has_dependents = st.selectbox("Has dependents", list(encoders["HasDependents"].classes_))

with tab2:
    loan_amount = st.number_input("Loan amount", 1000, 250000, 20000, step=1000)
    loan_term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60])
    loan_purpose = st.selectbox("Loan purpose", list(encoders["LoanPurpose"].classes_))
    interest_rate = st.slider("Interest rate (%)", 1.0, 30.0, 12.0, step=0.1)
    has_cosigner = st.selectbox("Has co-signer", list(encoders["HasCoSigner"].classes_))
    has_mortgage = st.selectbox("Has mortgage", list(encoders["HasMortgage"].classes_))

with tab3:
    income = st.number_input("Annual income", 10000, 200000, 60000, step=1000)
    credit_score = st.slider("Credit score", 300, 850, 650)
    months_employed = st.slider("Months employed", 0, 480, 60)
    num_credit_lines = st.slider("Number of credit lines", 0, 20, 3)
    dti_ratio = st.slider("Debt-to-income ratio", 0.0, 1.0, 0.35, step=0.01)

st.divider()
audience = st.radio("Explanation audience", ["Customer", "Analyst", "Customer + Analyst"], horizontal=True)
predict_btn = st.button("🔍 Predict default risk", type="primary", use_container_width=True)

if predict_btn:
    raw = {
        "Age": age, "Income": income, "LoanAmount": loan_amount,
        "CreditScore": credit_score, "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
        "LoanTerm": loan_term, "DTIRatio": dti_ratio,
        "Education": education, "EmploymentType": employment_type,
        "MaritalStatus": marital_status, "HasMortgage": has_mortgage,
        "HasDependents": has_dependents, "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
    }
    row = {}
    for col in feature_names:
        val = raw[col]
        if col in encoders:
            val = encoders[col].transform([val])[0]
        row[col] = val
    X_input = pd.DataFrame([row])[feature_names]

    default_probability = float(model.predict_proba(X_input)[0, 1])
    prediction = "DEFAULT" if default_probability >= threshold else "NO DEFAULT"
    risk, banner_bg, banner_text, icon, message = risk_tier(default_probability)

    shap_values = explainer.shap_values(X_input)
    base_value = float(explainer.expected_value)
    contributions = sorted(
        zip(feature_names, shap_values[0]), key=lambda x: abs(x[1]), reverse=True
    )[:8]

    # persist everything downstream pages need
    st.session_state["prediction_ready"] = True
    st.session_state["raw_inputs"] = raw
    st.session_state["X_input"] = X_input
    st.session_state["default_probability"] = default_probability
    st.session_state["prediction"] = prediction
    st.session_state["risk"] = risk
    st.session_state["banner_bg"] = banner_bg
    st.session_state["banner_text"] = banner_text
    st.session_state["icon"] = icon
    st.session_state["message"] = message
    st.session_state["contributions"] = [(f, float(v)) for f, v in contributions]
    st.session_state["base_value"] = base_value
    st.session_state["threshold"] = threshold
    st.session_state["audience"] = audience

# NOTE: this block reads session_state (persists across reruns), not predict_btn
# (which resets to False on every rerun, including the one triggered by clicking
# a button below — that was the original bug: buttons appeared but never fired).
if st.session_state.get("prediction_ready"):
    st.success("Prediction complete — see the Results, SHAP Explainability, and AI Explanation pages.")
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("View results  →", use_container_width=True):
            st.switch_page("pages_app/results.py")
    with nav2:
        if st.button("View SHAP explainability  →", use_container_width=True):
            st.switch_page("pages_app/shap_explainability.py")
    with nav3:
        if st.button("View AI explanation  →", use_container_width=True):
            st.switch_page("pages_app/ai_explanation.py")
