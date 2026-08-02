import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from common import load_artifacts

st.title("📈 SHAP explainability")
st.caption("Understand which factors influenced this prediction.")

if not st.session_state.get("prediction_ready"):
    st.info("No prediction yet. Go to the **Prediction** page and submit an applicant first.")
    if st.button("Go to Prediction  →"):
        st.switch_page("pages_app/prediction.py")
    st.stop()

_, encoders, _, _ = load_artifacts()
contributions = st.session_state["contributions"]
X_input = st.session_state["X_input"]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Feature contributions")
    feats = [c[0] for c in contributions][::-1]
    vals = [c[1] for c in contributions][::-1]
    bar_colors = ["#2E5BFF" if v > 0 else "#94A3B8" for v in vals]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feats, vals, color=bar_colors)
    ax.set_xlabel("SHAP impact (gray reduces risk | blue increases risk)")
    ax.axvline(0, color="black", linewidth=0.8)
    st.pyplot(fig)

with col2:
    st.subheader("Applicant values")
    rows = []
    for feature, impact in contributions:
        value = (
            X_input.iloc[0][feature] if feature not in encoders
            else encoders[feature].inverse_transform([int(X_input.iloc[0][feature])])[0]
        )
        rows.append({"Feature": feature, "Value": value, "SHAP impact": round(impact, 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
if st.button("See AI explanation  →"):
    st.switch_page("pages_app/ai_explanation.py")
