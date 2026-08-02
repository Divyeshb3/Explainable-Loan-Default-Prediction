"""
common.py — shared resources used across all pages of the multi-page app.
"""
import json
import tempfile
from datetime import datetime

import joblib
import streamlit as st
import xgboost as xgb
import shap
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

# ---------------------------------------------------------------------------
# Theme — navy & blue (matches reference dashboard)
# ---------------------------------------------------------------------------
THEME = {
    "primary": "#2E5BFF",
    "primary_dark": "#1B3FCC",
    "navy": "#0F1B3D",
    "gray": "#94A3B8",
    "gray_dark": "#1E293B",
    "low_bg": "#E1F3E5", "low_text": "#166534",
    "medium_bg": "#FEF3D6", "medium_text": "#92610A",
    "high_bg": "#FDE1E1", "high_text": "#B91C1C",
    "surface": "#F5F7FB",
    "customer_bg": "#EAF0FF", "customer_accent": "#2E5BFF", "customer_text": "#1B3FCC",
    "analyst_bg": "#F1F5F9", "analyst_accent": "#475569", "analyst_text": "#334155",
}


# ---------------------------------------------------------------------------
# Cached model artifacts (shared across every page in the session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_model.json")
    encoders = joblib.load("models/encoders.pkl")
    with open("models/model_metadata.json") as f:
        metadata = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, encoders, metadata, explainer


# ---------------------------------------------------------------------------
# Risk tier helper (single source of truth, used by Prediction + Results + AI pages)
# ---------------------------------------------------------------------------
def risk_tier(default_probability):
    if default_probability < 0.30:
        return "Low Risk", THEME["low_bg"], THEME["low_text"], "🟢", \
            "Applicant has a low probability of default."
    elif default_probability < 0.60:
        return "Medium Risk", THEME["medium_bg"], THEME["medium_text"], "🟡", \
            "Applicant has a moderate probability of default."
    else:
        return "High Risk", THEME["high_bg"], THEME["high_text"], "🔴", \
            "Applicant has a high probability of default."


# ---------------------------------------------------------------------------
# Verdict banner + threshold gauge (reused on Results + SHAP pages)
# ---------------------------------------------------------------------------
def render_verdict_banner(risk, banner_bg, banner_text, icon, message, probability):
    st.markdown(f"""
    <div style="background:{banner_bg}; border-radius:12px; padding:20px 24px;
                display:flex; align-items:center; justify-content:space-between;
                flex-wrap:wrap; gap:16px; margin-bottom:6px;">
      <div style="display:flex; align-items:center; gap:14px;">
        <span style="font-size:28px;">{icon}</span>
        <div>
          <div style="font-size:20px; font-weight:700; color:{banner_text};">{risk}</div>
          <div style="font-size:13px; color:{banner_text}; opacity:0.85;">{message}</div>
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:30px; font-weight:700; color:{banner_text};">{probability:.1%}</div>
        <div style="font-size:12px; color:{banner_text}; opacity:0.85;">predicted default probability</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_threshold_gauge(probability, threshold, prediction):
    prob_pct = min(max(probability * 100, 0), 100)
    thresh_pct = min(max(threshold * 100, 0), 100)
    st.markdown(f"""
    <div style="position:relative; margin:16px 4px 28px;">
      <div style="height:8px; border-radius:4px;
                  background:linear-gradient(90deg,#94A3B8 0%, #2E5BFF 55%, #B91C1C 100%);
                  opacity:0.4;"></div>
      <div style="position:absolute; top:-5px; left:{prob_pct}%; width:2px; height:18px; background:#0F1B3D;"></div>
      <div style="position:absolute; top:-5px; left:{thresh_pct}%; width:2px; height:18px;
                  border-left:1px dashed #475569;"></div>
      <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:12px; color:#475569;">
        <span>0%</span>
        <span>probability {probability:.1%} · threshold {threshold:.1%} ·
              prediction: <strong style="color:#0F1B3D;">{prediction.lower()}</strong></span>
        <span>100%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def explanation_card(text, audience):
    accent = THEME["customer_accent"] if audience == "customer" else THEME["analyst_accent"]
    bg = THEME["customer_bg"] if audience == "customer" else THEME["analyst_bg"]
    text_color = THEME["customer_text"] if audience == "customer" else THEME["analyst_text"]
    st.markdown(
        f'<div style="border-left:3px solid {accent}; background:{bg}; '
        f'border-radius:0 8px 8px 0; padding:14px 16px; font-size:14px; color:{text_color};">'
        f'{text}</div>', unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Hero graphic — flat SVG illustration, coral/gray theme, no stock imagery
# ---------------------------------------------------------------------------
def hero_svg():
    return """
    <svg width="100%" viewBox="0 0 480 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Risk gauge, shield and rising bar chart illustration">
      <circle cx="240" cy="170" r="150" fill="#F5F7FB"/>
      <path d="M 110 200 A 130 130 0 0 1 370 200" fill="none" stroke="#E2E8F0" stroke-width="26" stroke-linecap="round"/>
      <path d="M 110 200 A 130 130 0 0 1 240 70" fill="none" stroke="#94A3B8" stroke-width="26" stroke-linecap="round"/>
      <path d="M 240 70 A 130 130 0 0 1 370 200" fill="none" stroke="#2E5BFF" stroke-width="26" stroke-linecap="round"/>
      <circle cx="240" cy="200" r="10" fill="#0F1B3D"/>
      <line x1="240" y1="200" x2="305" y2="130" stroke="#0F1B3D" stroke-width="5" stroke-linecap="round"/>
      <g transform="translate(150,235)">
        <rect x="0" y="24" width="26" height="26" rx="3" fill="#CBD5E1"/>
        <rect x="34" y="10" width="26" height="40" rx="3" fill="#94A3B8"/>
        <rect x="68" y="0" width="26" height="50" rx="3" fill="#2E5BFF"/>
        <rect x="102" y="16" width="26" height="34" rx="3" fill="#475569"/>
      </g>
      <g transform="translate(300,215)">
        <rect x="0" y="0" width="80" height="90" rx="14" fill="#0F1B3D"/>
        <path d="M 40 20 L 66 32 L 66 55 C 66 70 54 78 40 82 C 26 78 14 70 14 55 L 14 32 Z" fill="#BFD0FF"/>
        <path d="M 30 51 L 37 58 L 51 43" fill="none" stroke="#0F1B3D" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      </g>
    </svg>
    """


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------
def _add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(inch, 11 * inch, "Loan Default Prediction Report")
    canvas.setFont("Helvetica", 9)
    canvas.drawString(inch, 10.8 * inch, "Explainable AI Risk Assessment")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(inch, 0.5 * inch, "Generated using XGBoost + SHAP + Groq LLaMA")
    canvas.drawRightString(7.5 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf(report_data):
    styles = getSampleStyleSheet()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name)
    story = []

    story.append(Paragraph("Loan Default Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}", styles["Normal"]
    ))
    story.append(Spacer(1, 18))

    story.append(Paragraph("<b>Prediction Summary</b>", styles["Heading2"]))
    prediction_table = [
        ["Item", "Value"],
        ["Prediction", report_data["Prediction"]],
        ["Risk Level", report_data["Risk Level"]],
        ["Default Probability", report_data["Default Probability"]],
        ["Decision Threshold", report_data["Decision Threshold"]],
    ]
    prediction_tbl = Table(prediction_table, colWidths=[180, 180])
    prediction_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(prediction_tbl)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Applicant Details</b>", styles["Heading2"]))
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
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Top Risk Factors</b>", styles["Heading2"]))
    risk_table = [["Feature", "SHAP Impact"]]
    for factor in report_data["Top Risk Factors"]:
        feature, impact = factor.split(":")
        risk_table.append([feature.strip(), impact.strip()])
    risk_tbl = Table(risk_table, colWidths=[220, 120])
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(risk_tbl)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>AI Explanation</b>", styles["Heading2"]))
    story.append(Paragraph(
        report_data["AI Explanation"].replace("\n", "<br/>"), styles["BodyText"]
    ))

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return temp_file.name
