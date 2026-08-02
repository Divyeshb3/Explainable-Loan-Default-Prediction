import streamlit as st
from common import generate_pdf

st.title("📄 Reports")
st.caption("Download a complete PDF summary of this prediction.")

if not st.session_state.get("prediction_ready"):
    st.info("No prediction yet. Go to the **Prediction** page and submit an applicant first.")
    if st.button("Go to Prediction  →"):
        st.switch_page("pages_app/prediction.py")
    st.stop()

customer_text = st.session_state.get("customer_text", "")
analyst_text = st.session_state.get("analyst_text", "")

if not customer_text and not analyst_text:
    st.warning("No AI explanation generated yet. Visit the AI Explanation page first for a complete report.")

ai_explanation_text = ""
if customer_text:
    ai_explanation_text += "Customer Explanation\n\n" + customer_text
if analyst_text:
    if ai_explanation_text:
        ai_explanation_text += "\n\n----------------------------------------\n\n"
    ai_explanation_text += "Analyst Explanation\n\n" + analyst_text

raw = st.session_state["raw_inputs"]
report_data = {
    "Prediction": st.session_state["prediction"],
    "Risk Level": st.session_state["risk"],
    "Default Probability": f"{st.session_state['default_probability']:.2%}",
    "Decision Threshold": f"{st.session_state['threshold']:.2%}",
    "Age": raw["Age"],
    "Annual Income": raw["Income"],
    "Loan Amount": raw["LoanAmount"],
    "Credit Score": raw["CreditScore"],
    "Interest Rate": f"{raw['InterestRate']}%",
    "Loan Term": raw["LoanTerm"],
    "Debt-to-Income Ratio": raw["DTIRatio"],
    "Top Risk Factors": [
        f"{feature}: {impact:.3f}" for feature, impact in st.session_state["contributions"][:5]
    ],
    "AI Explanation": ai_explanation_text or "No explanation generated.",
}

st.subheader("Report includes")
st.markdown(
    "- Applicant information\n"
    "- Prediction summary\n"
    "- Top SHAP risk factors\n"
    "- AI-generated explanation\n"
)

pdf_path = generate_pdf(report_data)
with open(pdf_path, "rb") as f:
    st.download_button(
        label="📄 Download prediction report",
        data=f,
        file_name="loan_default_prediction_report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
