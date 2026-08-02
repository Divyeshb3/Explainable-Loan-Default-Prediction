import json
import streamlit as st

st.title("ℹ️ About this project")

st.markdown("""
### Explainable Loan Default Prediction

This tool predicts loan default risk using an XGBoost classifier, explains individual
predictions using SHAP, and translates those explanations into natural language using
Groq-hosted LLaMA 3.3 70B — one version for the applicant, one for a risk analyst.
""")

with open("models/model_metadata.json") as f:
    metadata = json.load(f)

st.subheader("Model performance")
c1, c2 = st.columns(2)
c1.metric("ROC-AUC", f"{metadata.get('auc', 0.758):.3f}")
c2.metric("Decision threshold (F1-optimal)", f"{metadata.get('best_threshold', 0.641):.3f}")

st.subheader("Tech stack")
st.markdown("""
- **Model:** XGBoost, trained on 255,347 loan applications with class-imbalance handling (`scale_pos_weight`)
- **Explainability:** SHAP (TreeExplainer) — per-prediction, not just global, feature attribution
- **LLM:** Groq-hosted LLaMA 3.3 70B, with separate customer/analyst prompts
- **Evaluation:** Independent LLM-as-judge scoring on faithfulness, direction correctness, clarity, and consistency
- **Interface:** Streamlit multi-page app
- **Reports:** ReportLab-generated PDF summaries
""")

st.subheader("Links")
st.markdown("""
- [GitHub repository](https://github.com/Divyeshb3/Explainable-Loan-Default-Prediction)
- [Dataset (Kaggle)](https://www.kaggle.com/datasets/nikhil1e9/loan-default)
""")
