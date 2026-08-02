import streamlit as st
from common import render_verdict_banner, render_threshold_gauge

st.title("📊 Prediction results")

if not st.session_state.get("prediction_ready"):
    st.info("No prediction yet. Go to the **Prediction** page and submit an applicant first.")
    if st.button("Go to Prediction  →"):
        st.switch_page("pages_app/prediction.py")
    st.stop()

render_verdict_banner(
    st.session_state["risk"], st.session_state["banner_bg"],
    st.session_state["banner_text"], st.session_state["icon"],
    st.session_state["message"], st.session_state["default_probability"],
)
render_threshold_gauge(
    st.session_state["default_probability"], st.session_state["threshold"],
    st.session_state["prediction"],
)

c1, c2, c3 = st.columns(3)
c1.metric("Prediction", st.session_state["prediction"])
c2.metric("Decision threshold", f"{st.session_state['threshold']:.1%}")
c3.metric("Default probability", f"{st.session_state['default_probability']:.1%}")

st.divider()
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("See SHAP explainability  →", use_container_width=True):
        st.switch_page("pages_app/shap_explainability.py")
with nav2:
    if st.button("See AI explanation  →", use_container_width=True):
        st.switch_page("pages_app/ai_explanation.py")
