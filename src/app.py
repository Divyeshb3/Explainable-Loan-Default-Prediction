import os
import json

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)   

from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import tempfile
from reportlab.lib.units import inch
from reportlab.lib import colors

from groq_explain import explain_with_llm

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
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

def add_page_number(canvas, doc):

    canvas.saveState()

    # Header
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(
        inch,
        11 * inch,
        "Loan Default Prediction Report"
    )

    canvas.setFont("Helvetica", 9)
    canvas.drawString(
        inch,
        10.8 * inch,
        "Explainable AI Risk Assessment"
    )

    # Footer
    canvas.setFont("Helvetica", 8)

    canvas.drawString(
        inch,
        0.5 * inch,
        "Generated using XGBoost + SHAP + Groq LLaMA"
    )

    canvas.drawRightString(
        7.5 * inch,
        0.5 * inch,
        f"Page {doc.page}"
    )

    canvas.restoreState()
    
def generate_pdf(report_data):

    styles = getSampleStyleSheet()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    doc = SimpleDocTemplate(temp_file.name)

    story = []

    # Title
    story.append(
        Paragraph(
            "Loan Default Prediction Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 18))

    # Prediction Summary
    story.append(
        Paragraph("<b>Prediction Summary</b>", styles["Heading2"])
    )

    prediction_table = [
        ["Item", "Value"],
        ["Prediction", report_data["Prediction"]],
        ["Risk Level", report_data["Risk Level"]],
        ["Default Probability", report_data["Default Probability"]],
        ["Decision Threshold", report_data["Decision Threshold"]],
    ]

    prediction_tbl = Table(prediction_table, colWidths=[180, 180])

    prediction_tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),

            ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),

            ("GRID", (0, 0), (-1, -1), 1, colors.black),

            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

            ("ALIGN", (0, 0), (-1, -1), "CENTER"),

            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(prediction_tbl)
    story.append(Spacer(1, 15))

    # Applicant Details
    story.append(
        Paragraph("<b>Applicant Details</b>", styles["Heading2"])
    )

    table_data = [
        ["Field", "Value"],
        ["Age", report_data["Age"]],
        ["Annual Income", report_data["Annual Income"]],
        ["Loan Amount", report_data["Loan Amount"]],
        ["Credit Score", report_data["Credit Score"]],
        ["Interest Rate", report_data["Interest Rate"]],
        ["Loan Term", report_data["Loan Term"]],
        ["Debt-to-Income Ratio", report_data["Debt-to-Income Ratio"]],
    ]

    table = Table(table_data, colWidths=[180, 180])

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),

            ("GRID", (0, 0), (-1, -1), 1, colors.black),

            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),

            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 1), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),

            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 15))


    # SHAP Factors
    story.append(
        Paragraph("<b>Top Risk Factors</b>", styles["Heading2"])
    )

    risk_table = [["Feature", "SHAP Impact"]]

    for factor in report_data["Top Risk Factors"]:

        feature, impact = factor.split(":")

        risk_table.append([
            feature.strip(),
            impact.strip()
        ])

    risk_tbl = Table(risk_table, colWidths=[220, 120])

    risk_tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),

            ("GRID", (0, 0), (-1, -1), 1, colors.black),

            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),

            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

            ("ALIGN", (0, 0), (-1, -1), "CENTER"),

            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(risk_tbl)
    story.append(Spacer(1, 15))


    story.append(
        Paragraph("<b>AI Explanation</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            report_data["AI Explanation"].replace("\n", "<br/>"),
            styles["BodyText"]
        )
    )
    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    return temp_file.name

feature_names = metadata["feature_names"]
threshold = metadata["best_threshold"]

st.title("🏦 Loan Default Predictor")

st.markdown("""
### Explainable AI for Loan Risk Assessment

Predict the probability of loan default using **XGBoost**, understand the prediction with **SHAP**, and generate audience-specific explanations using **Groq LLaMA 3.3 70B**.
""")

# Model Information
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Model", "XGBoost")

with c2:
    st.metric("Explainability", "SHAP")

with c3:
    st.metric("LLM", "Groq LLaMA")

st.divider()

# ---------- Sidebar input form ----------
st.sidebar.title("Applicant Information")

st.sidebar.markdown(
    "Enter the applicant's details and click **Predict**."
)

st.sidebar.divider()

def cat_input(label, col):
    options = list(encoders[col].classes_)
    return st.sidebar.selectbox(label, options)

st.sidebar.subheader("Financial Details")

age = st.sidebar.slider("Age", 18, 75, 35)
income = st.sidebar.number_input("Annual Income", 10000, 200000, 60000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 250000, 20000, step=1000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
months_employed = st.sidebar.slider("Months Employed", 0, 480, 60)
num_credit_lines = st.sidebar.slider("Number of Credit Lines", 0, 20, 3)
interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 30.0, 12.0, step=0.1)
loan_term = st.sidebar.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
dti_ratio = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, step=0.01)

st.sidebar.divider()
st.sidebar.subheader("Applicant Profile")

education = cat_input("Education", "Education")
employment_type = cat_input("Employment Type", "EmploymentType")
marital_status = cat_input("Marital Status", "MaritalStatus")
has_mortgage = cat_input("Has Mortgage", "HasMortgage")
has_dependents = cat_input("Has Dependents", "HasDependents")
loan_purpose = cat_input("Loan Purpose", "LoanPurpose")
has_cosigner = cat_input("Has Co-Signer", "HasCoSigner")

st.sidebar.divider()
st.sidebar.subheader("Explanation")

audience = st.sidebar.radio(
    "Audience",
    ["Customer",
     "Analyst",
     "Customer + Analyst"
     ]
)

predict_btn = st.sidebar.button(
    "🔍 Predict Default Risk",
    type="primary",
    use_container_width=True
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

    st.header("📊 Prediction Results")

    X_input = build_input_row()

    default_probability = model.predict_proba(X_input)[0, 1]
    prediction = "DEFAULT" if default_probability >= threshold else "NO DEFAULT"

    col1, col2 = st.columns([1, 1])

    with col1:

        if default_probability < 0.30:
            risk = "🟢 Low Risk"
            box = st.success
            message = "Applicant has a low probability of default."

        elif default_probability < 0.60:
            risk = "🟡 Medium Risk"
            box = st.warning
            message = "Applicant has a moderate probability of default."

        else:
            risk = "🔴 High Risk"
            box = st.error
            message = "Applicant has a high probability of default."

        box(f"""
        ### {risk}

        {message}
        """)

        st.metric(
            "Default Probability",
            f"{default_probability:.1%}"
        )

        st.metric(
            "Decision Threshold",
            f"{threshold:.1%}"
        )

        st.metric(
            "Prediction",
            prediction
        )

        st.progress(float(default_probability))
        
        
              
# ---------- SHAP for this single row ----------
    shap_values = explainer.shap_values(X_input)
    base_value = explainer.expected_value

    contributions = sorted(
        zip(feature_names, shap_values[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:8]
    positive = [c for c in contributions if c[1] > 0]
    negative = [c for c in contributions if c[1] < 0]

    with col2:
        
        st.subheader("📈 SHAP Feature Contributions")
        
        shap_fig, shap_ax = plt.subplots(figsize=(6, 4))

        feats = [c[0] for c in contributions][::-1]
        vals = [c[1] for c in contributions][::-1]

        bar_colors = [
            "#d62728" if v > 0 else "#2ca02c"
            for v in vals
        ]

        shap_ax.barh(feats, vals, color=bar_colors)
        shap_ax.set_xlabel(
            "SHAP impact (→ increases risk | decreases risk ←)"
        )
        shap_ax.axvline(0, color="black", linewidth=0.8)

        st.pyplot(shap_fig)
        st.caption(
            "Positive SHAP values increase the predicted risk of default, while negative SHAP values reduce it."
        )
        

    st.divider()

    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("📋 Feature Contribution Details")

        feature_rows = []

        for feature, impact in contributions:

            value = (
                X_input.iloc[0][feature]
                if feature not in encoders
                else encoders[feature].inverse_transform(
                    [int(X_input.iloc[0][feature])]
                )[0]
            )

            feature_rows.append({
                "Feature": feature,
                "Value": value,
                "SHAP Impact": round(float(impact), 3)
            })

        st.dataframe(
            pd.DataFrame(feature_rows),
            use_container_width=True,
            hide_index=True
        )
    
    with col4:

        st.subheader("🔴 Top Risk Increasing Factors")

        if positive:
            for feature, impact in positive:
                st.write(f"**{feature}** : +{impact:.3f}")
        else:
            st.write("None")

        st.subheader("🟢 Top Risk Reducing Factors")

        if negative:
            for feature, impact in negative:
                st.write(f"**{feature}** : {impact:.3f}")
        else:
            st.write("None")
        
    # ---------- Build explanation dict for LLM ----------
    explanation = {
        "predicted_default_probability": float(default_probability),
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
    
    report_data = {
        "Prediction": prediction,
        "Risk Level": risk,
        "Default Probability": f"{default_probability:.2%}",
        "Decision Threshold": f"{threshold:.2%}",

        "Age": age,
        "Annual Income": income,
        "Loan Amount": loan_amount,
        "Credit Score": credit_score,
        "Months Employed": months_employed,
        "Interest Rate": f"{interest_rate}%",
        "Loan Term": loan_term,
        "Debt-to-Income Ratio": dti_ratio,

        "Top Risk Factors": [
            f"{feature}: {impact:.3f}"
            for feature, impact in contributions[:5]
        ],
        "AI Explanation": ""
    }

    st.header("🤖 AI Explanation")

    if os.environ.get("GROQ_API_KEY"):

        if audience == "Customer + Analyst":

            # Customer Explanation
            st.subheader("👤 Customer Explanation")

            with st.spinner("Generating customer explanation..."):
                customer_text = explain_with_llm(
                    explanation,
                    audience="customer"
                )

            with st.container(border=True):
                st.markdown(customer_text)

            st.divider()

            # Analyst Explanation
            st.subheader("📊 Analyst Explanation")

            with st.spinner("Generating analyst explanation..."):
                analyst_text = explain_with_llm(
                    explanation,
                    audience="analyst"
                )
            report_data["AI Explanation"] = (
                "Customer Explanation\n\n"
                + customer_text
                + "\n\n----------------------------------------\n\n"
                + "Analyst Explanation\n\n"
                + analyst_text
            )

            with st.container(border=True):
                st.markdown(analyst_text)
                      
        else:

            st.info(f"Audience: **{audience}**")

            with st.spinner("Generating AI explanation..."):
                try:
                    narration = explain_with_llm(
                        explanation,
                        audience=audience.lower()
                    )
                    report_data["AI Explanation"] = narration

                    with st.container(border=True):
                        st.markdown(narration)

                except Exception as e:
                    st.error(
                        "Unable to generate the explanation. Please check your Groq API key or network connection."
                    )

                    with st.expander("Technical Details"):
                        st.code(str(e))
        # Download Report
        pdf_path = generate_pdf(report_data)

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Prediction Report",
                data=pdf_file,
                file_name="loan_default_prediction_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.warning(
            "Groq API key not found. Configure the GROQ_API_KEY environment variable to enable AI-generated explanations."
        )


        with st.expander("Raw SHAP contribution data"):
            st.json(explanation)
            

else:
    st.info(
        """
        ### Getting Started

        1. Enter the applicant's information in the sidebar.
        2. Click **Predict Default Risk**.
        3. Review:
        - Default probability
        - SHAP feature contributions
        - LLM-generated explanation
        """
        )