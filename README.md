# Loan Default Predictor — Explainable AI + LLM Narration

Predicts loan default risk with XGBoost, explains *why* using SHAP, and translates
those SHAP values into plain-English explanations using an LLM (Groq / LLaMA 3.3 70B) —
one version for the applicant, one for a risk analyst, plus a combined view. Includes
color-coded risk tiers, a downloadable PDF prediction report, and an independently
evaluated explanation-quality score.

**Live demo:** [Streamlit Community Cloud](https://explainable-loan-default-prediction-kjbtysmspmtadqzuf8ude7.streamlit.app/)

## Features
- XGBoost default-risk prediction with class-imbalance handling
- Per-prediction SHAP explainability (not just global feature importance)
- Dual-audience LLM narration: customer, analyst, or both side-by-side
- Color-coded risk tiers (🟢 low / 🟡 medium / 🔴 high)
- Downloadable PDF prediction report (ReportLab)
- Independent LLM-as-judge evaluation of explanation quality

## Dataset
`data/Loan_default.csv` — 255,347 rows, 18 columns (Kaggle: nikhil1e9/loan-default).

## Setup
```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
```

## Pipeline (run in order)
```bash
python3 train_baseline.py   # trains XGBoost, saves model + encoders
python3 shap_explain.py     # computes global + example SHAP explanations
python3 groq_explain.py     # demo: LLM narration for one example (CLI)
streamlit run app.py        # interactive app
```

## Results
- ROC-AUC: 0.758
- Tuned-threshold (F1-optimal, 0.641): 46% recall / 31% precision on the default class
- Top global drivers: Age, InterestRate, MonthsEmployed, Income, LoanAmount

## Model Selection & Evaluation

**Why AUC and recall, not accuracy:** the dataset has an 11.6% default rate, so a model
that predicts "No Default" for every applicant — doing zero real work — already scores
88.4% accuracy. Optimizing for accuracy directly would push the model toward flagging
fewer defaulters, not more. For that reason, this project reports ROC-AUC (threshold-
independent) and recall/precision on the default class instead, and treats the
classification threshold as a deliberate business tradeoff rather than a fixed default.

| Threshold | Accuracy | Recall (Default) | Precision (Default) |
|---|---|---|---|
| 0.50 | 69.8% | 68.3% | 23.0% |
| 0.641 (F1-optimal) | ~79% | 46.5% | 31.1% |

A bank prioritizing catching defaulters would run at threshold 0.5 (higher recall, more
false alarms); a bank prioritizing not rejecting good applicants would use the tuned
threshold (fewer false alarms, more missed defaulters).

**Feature engineering & tuning investigated:** derived features (loan-to-income ratio,
debt burden, employment stability, age binning) and Optuna hyperparameter search were
tested to see if they could raise the AUC above baseline. Result: AUC stayed essentially
flat (0.757 vs 0.758 baseline), indicating the gradient-boosted model already captures
these feature interactions implicitly from the raw columns, and that ~0.76 AUC is close
to this dataset's real predictive ceiling — a known characteristic also seen in other
public analyses of this dataset. The original baseline model was kept as the deployed
model on that basis.

## Explainability Evaluation

Explanation quality was measured using an independent LLM-as-judge (Groq LLaMA 3.3 70B,
separate from the generation call to avoid self-evaluation bias — same methodology used
in the [MindCare AI project](https://github.com/Divyeshb3/mental-health-chatbot)),
scoring 25 stratified test cases (low/medium/high predicted risk) across both audience
modes on faithfulness, direction correctness, clarity, and consistency.

| Metric | Customer explanation | Analyst explanation | Overall |
|---|---|---|---|
| Faithfulness | 0.789 | 1.000 | 0.894 |
| Direction correctness | 0.904 | 0.993 | 0.949 |
| Clarity | 0.896 | 0.996 | 0.946 |
| Consistency | 1.000 | 1.000 | 1.000 |

**Key finding:** after a fix grounding the LLM's stated verdict in the model's actual
output (rather than letting it infer a verdict from the SHAP factor list alone),
analyst-facing faithfulness reached a perfect 1.000 and direction correctness stayed
near-perfect (0.993). Consistency remained perfect (1.000) across all 50 evaluations
both before and after the fix — no internal contradictions were found. Customer-facing
faithfulness (0.789) continues to reflect deliberate omission rather than fabrication:
the customer prompt is constrained to 3-4 plain-language sentences, so it selectively
surfaces the top factors rather than every SHAP contributor, by design.

## Architecture
```
Loan_default.csv → XGBoost classifier → SHAP TreeExplainer
                                              ↓
                          per-prediction feature contributions
                                              ↓
                    Groq (LLaMA 3.3 70B) → plain-English explanation
                          (customer mode / analyst mode)
                                              ↓
                              Streamlit interface
```

## Files
- `train_baseline.py` — data prep + XGBoost training
- `shap_explain.py` — global importance + per-row local explanations
- `groq_explain.py` — LLM narration layer (two audience modes)
- `app.py` — Streamlit UI tying it all together
- `model_metadata.json`, `encoders.pkl`, `xgb_model.json` — trained artifacts

## Notes
- Class imbalance (~11.6% default rate) handled via `scale_pos_weight` in XGBoost.
- Categorical features label-encoded; encoders saved for consistent inference.
- SHAP uses `TreeExplainer` (exact, fast for tree models — no approximation needed).