import os
import streamlit as st
from common import explanation_card
from groq_explain import explain_with_llm

st.title("🤖 AI explanation")
st.caption("AI-generated explanation for this prediction, powered by Groq LLaMA 3.3 70B.")

if not st.session_state.get("prediction_ready"):
    st.info("No prediction yet. Go to the **Prediction** page and submit an applicant first.")
    if st.button("Go to Prediction  →"):
        st.switch_page("pages_app/prediction.py")
    st.stop()

if not os.environ.get("GROQ_API_KEY"):
    st.warning("Groq API key not found. Set the GROQ_API_KEY environment variable to enable AI-generated explanations.")
    with st.expander("Raw SHAP contribution data"):
        st.json(st.session_state["contributions"])
    st.stop()

explanation = {
    "predicted_default_probability": st.session_state["default_probability"],
    "actual_risk_tier": st.session_state["risk"],
    "actual_prediction": st.session_state["prediction"],
    "base_value": st.session_state["base_value"],
    "top_contributions": [
        {
            "feature": f,
            "value": st.session_state["raw_inputs"].get(f, "N/A"),
            "shap_impact": v,
        }
        for f, v in st.session_state["contributions"][:6]
    ],
}

audience = st.session_state.get("audience", "Customer + Analyst")
st.info(f"Audience selected on the Prediction page: **{audience}**")

customer_text, analyst_text = "", ""

if audience == "Customer + Analyst":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**👤 Customer explanation**")
        try:
            with st.spinner("Generating customer explanation..."):
                customer_text = explain_with_llm(explanation, audience="customer")
            explanation_card(customer_text, "customer")
        except Exception as e:
            st.error("Unable to generate the customer explanation. Please check your Groq API key or network connection.")
            with st.expander("Technical details"):
                st.code(str(e))
    with col2:
        st.markdown("**📊 Analyst explanation**")
        try:
            with st.spinner("Generating analyst explanation..."):
                analyst_text = explain_with_llm(explanation, audience="analyst")
            explanation_card(analyst_text, "analyst")
        except Exception as e:
            st.error("Unable to generate the analyst explanation. Please check your Groq API key or network connection.")
            with st.expander("Technical details"):
                st.code(str(e))
else:
    key = audience.lower()
    try:
        with st.spinner("Generating AI explanation..."):
            narration = explain_with_llm(explanation, audience=key)
        explanation_card(narration, key)
        if key == "customer":
            customer_text = narration
        else:
            analyst_text = narration
    except Exception as e:
        st.error("Unable to generate the explanation. Please check your Groq API key or network connection.")
        with st.expander("Technical details"):
            st.code(str(e))

st.session_state["customer_text"] = customer_text
st.session_state["analyst_text"] = analyst_text

st.divider()
if st.button("Download PDF report  →"):
    st.switch_page("pages_app/reports.py")
