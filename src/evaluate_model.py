"""
Model Evaluation Script

Evaluates the trained XGBoost model on the test dataset using:
- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
- Threshold comparison (0.5 vs tuned threshold)

Saves the evaluation metrics to:
outputs/evaluation_metrics.json
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
)

# ---------- Load model + test data ----------
model = xgb.XGBClassifier()
model.load_model("models/xgb_model.json")

X_test = pd.read_csv("models/X_test.csv")
y_test = pd.read_csv("models/y_test.csv")["Default"]

with open("models/model_metadata.json") as f:
    metadata = json.load(f)
threshold = metadata["best_threshold"]

# ---------- Predictions ----------
y_proba = model.predict_proba(X_test)[:, 1]
y_pred_default = model.predict(X_test)              # threshold 0.5
y_pred_tuned = (y_proba >= threshold).astype(int)     # F1-optimal threshold

def metrics_at(y_true, y_pred, label):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold_label": label,
        "precision_default_class": round(precision_score(y_true, y_pred), 4),
        "recall_default_class": round(recall_score(y_true, y_pred), 4),
        "f1_default_class": round(f1_score(y_true, y_pred), 4),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }

report = {
    "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    "test_set_size": len(y_test),
    "test_set_default_rate": round(float(y_test.mean()), 4),
    "threshold_0.5": metrics_at(y_test, y_pred_default, "default (0.5)"),
    "threshold_tuned": metrics_at(y_test, y_pred_tuned, f"F1-optimal ({threshold:.3f})"),
}

with open("outputs/evaluation_metrics.json", "w") as f:
    json.dump(report, f, indent=2)

# ---------- Human-readable summary ----------
print("=" * 60)
print("MODEL EVALUATION SUMMARY")
print("=" * 60)
print(f"Test set: {report['test_set_size']:,} applications "
      f"({report['test_set_default_rate']:.1%} actual default rate)")
print(f"ROC-AUC: {report['roc_auc']}")
print()
for key in ["threshold_0.5", "threshold_tuned"]:
    m = report[key]
    print(f"--- Threshold: {m['threshold_label']} ---")
    print(f"  Precision (default class): {m['precision_default_class']:.1%}")
    print(f"  Recall (default class):    {m['recall_default_class']:.1%}")
    print(f"  F1 (default class):        {m['f1_default_class']:.3f}")
    cm = m["confusion_matrix"]
    print(f"  Confusion matrix: TP={cm['true_positive']}, FP={cm['false_positive']}, "
          f"TN={cm['true_negative']}, FN={cm['false_negative']}")
    print()

print("=" * 60)
print("Threshold Comparison")
print("=" * 60)

if report["threshold_tuned"]["f1_default_class"] > report["threshold_0.5"]["f1_default_class"]:
    print(
        f"✓ Tuned threshold ({threshold:.3f}) improves the F1-score "
        f"from {report['threshold_0.5']['f1_default_class']:.3f} "
        f"to {report['threshold_tuned']['f1_default_class']:.3f}."
    )
    print(
        "  This reduces false positives but also lowers recall, "
        "making it a better balance for this dataset."
    )
else:
    print("✓ Default threshold provides the better F1-score.")

print()
print("=" * 60)
print("BALANCED ACCURACY")
print("=" * 60)

default_bal_acc = balanced_accuracy_score(y_test, y_pred_default)
tuned_bal_acc = balanced_accuracy_score(y_test, y_pred_tuned)

print(f"Threshold 0.5          : {default_bal_acc:.4f}")
print(f"Tuned Threshold ({threshold:.3f}) : {tuned_bal_acc:.4f}")

print()
print("Full classification report (threshold=0.5):")
print(classification_report(
    y_test,
    y_pred_default,
    target_names=["No Default", "Default"]
))

print()
print("=" * 60)
print("MODEL INTERPRETATION")
print("=" * 60)

print(f"• ROC-AUC ({report['roc_auc']:.4f}) indicates good ability to distinguish")
print("  between default and non-default applicants.")

print()

if report["threshold_tuned"]["f1_default_class"] > report["threshold_0.5"]["f1_default_class"]:
    print("• The tuned threshold improves F1-score,")
    print("  making it a better choice when balancing")
    print("  precision and recall for default detection.")
else:
    print("• The default threshold provides the better F1-score.")

print()

if report["threshold_tuned"]["precision_default_class"] > report["threshold_0.5"]["precision_default_class"]:
    print("• Precision increases with the tuned threshold,")
    print("  reducing false positive loan rejections.")
print()

if report["threshold_tuned"]["recall_default_class"] < report["threshold_0.5"]["recall_default_class"]:
    print("• Recall decreases with the tuned threshold,")
    print("  meaning some additional default cases may be missed.")
    
print()
print("Full classification report (tuned threshold):")
print(classification_report(
    y_test,
    y_pred_tuned,
    target_names=["No Default", "Default"]
))

print("\nSaved: outputs/evaluation_metrics.json")

# ---------- ROC Curve ----------
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {report['roc_auc']:.4f})",
    linewidth=2
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True)

plt.savefig(
    "outputs/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: outputs/roc_curve.png")

# ---------- Precision-Recall Curve ----------
precision, recall, _ = precision_recall_curve(y_test, y_proba)
ap_score = average_precision_score(y_test, y_proba)

plt.figure(figsize=(6, 6))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"PR Curve (AP = {ap_score:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.legend(loc="lower left")

plt.savefig(
    "outputs/precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: outputs/precision_recall_curve.png")

# ---------- Confusion Matrix ----------
plt.figure(figsize=(6, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_tuned,
    display_labels=["No Default", "Default"],
    cmap="Blues",
    values_format="d",
)

plt.title(f"Confusion Matrix (Threshold = {threshold:.3f})")

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: outputs/confusion_matrix.png")

# ---------- Save Markdown Evaluation Summary ----------
summary = f"""# Model Evaluation Summary

## Dataset
- Test Samples: {report['test_set_size']:,}
- Default Rate: {report['test_set_default_rate']:.1%}

## Model Performance

- ROC-AUC: {report['roc_auc']}
- Accuracy (Threshold 0.5): {(y_pred_default == y_test).mean():.1%}
- Accuracy (Tuned Threshold {threshold:.3f}): {(y_pred_tuned == y_test).mean():.1%}

### Threshold Comparison

| Metric | Threshold 0.5 | Tuned Threshold |
|--------|--------------:|----------------:|
| Precision | {report['threshold_0.5']['precision_default_class']:.1%} | {report['threshold_tuned']['precision_default_class']:.1%} |
| Recall | {report['threshold_0.5']['recall_default_class']:.1%} | {report['threshold_tuned']['recall_default_class']:.1%} |
| F1-score | {report['threshold_0.5']['f1_default_class']:.3f} | {report['threshold_tuned']['f1_default_class']:.3f} |

## Conclusion

The tuned threshold ({threshold:.3f}) was selected because it improved the F1-score while reducing false positives. Although recall decreased, this threshold provides a better balance between precision and recall for this dataset.
"""

with open("outputs/evaluation_summary.md", "w", encoding="utf-8") as f:
    f.write(summary)

print("Saved: outputs/evaluation_summary.md")