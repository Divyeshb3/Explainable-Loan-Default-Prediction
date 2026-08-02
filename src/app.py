"""
app.py — entry point. Defines the multi-page navigation structure.

"""
import streamlit as st

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0F1B3D;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebarNav"] a {
    border-radius: 8px;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: #2E5BFF;
}
[data-testid="stSidebarNav"] a[aria-current="page"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebarNav"] a:hover {
    background-color: #1B2A54;
}
</style>
""", unsafe_allow_html=True)

pages = [
    st.Page("pages_app/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages_app/prediction.py", title="Prediction", icon="📝"),
    st.Page("pages_app/results.py", title="Results", icon="📊"),
    st.Page("pages_app/shap_explainability.py", title="SHAP Explainability", icon="📈"),
    st.Page("pages_app/ai_explanation.py", title="AI Explanation", icon="🤖"),
    st.Page("pages_app/reports.py", title="Reports", icon="📄"),
    st.Page("pages_app/about.py", title="About", icon="ℹ️"),
]

nav = st.navigation(pages)
nav.run()
