import json
import streamlit as st
from common import hero_svg

with open("models/model_metadata.json") as f:
    metadata = json.load(f)
auc = metadata.get("auc", 0.758)

col_text, col_hero = st.columns([1.1, 1])

with col_text:
    st.markdown(
        '<div style="display:inline-block; background:#EAF0FF; color:#1B3FCC; '
        'font-size:12px; font-weight:500; padding:5px 12px; border-radius:20px; margin-bottom:14px;">'
        'AI-powered credit risk assessment</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:38px; font-weight:700; line-height:1.2; color:#0F1B3D;">'
        'Explainable Loan Default <span style="color:#2E5BFF;">Prediction</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:15px; color:#475569; margin-top:14px; max-width:480px;">'
        'Predict the likelihood of loan default with machine learning, understand '
        'every prediction with SHAP explainability, and read AI-generated, '
        'audience-specific insights.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    if st.button("Make a prediction  →", type="primary"):
        st.switch_page("pages_app/prediction.py")

with col_hero:
    st.markdown(hero_svg(), unsafe_allow_html=True)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "🧩", "ML model", "XGBoost classifier trained on 255K+ loan records"),
    (c2, "🔍", "Explainable AI", "SHAP values show feature impact transparently"),
    (c3, "💬", "AI insights", "Natural language explanations using Groq LLaMA"),
    (c4, "📄", "PDF reports", "Download complete prediction reports"),
]
for col, icon, title, desc in cards:
    with col:
        st.markdown(
            f'<div style="background:#F5F7FB; border:1px solid #E2E8F0; border-radius:12px; padding:16px; height:150px;">'
            f'<div style="font-size:22px;">{icon}</div>'
            f'<div style="font-weight:600; color:#0F1B3D; margin-top:8px; font-size:14px;">{title}</div>'
            f'<div style="font-size:12px; color:#475569; margin-top:4px;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
st.metric("Model performance — ROC-AUC", f"{auc:.3f}")
st.caption(
    "Trained with class-imbalance handling on an 11.6% default-rate dataset. "
    "See the About page for the full evaluation methodology."
)
