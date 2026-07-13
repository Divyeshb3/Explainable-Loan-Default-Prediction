
import json
from groq_explain import explain_with_llm

# Load SHAP examples
with open("outputs/shap_examples.json") as f:
    examples = json.load(f)

for i, ex in enumerate(examples):

    label = (
        "ACTUAL DEFAULT"
        if ex["actual_label"] == 1
        else "ACTUAL NO DEFAULT"
    )

    print("\n" + "=" * 70)
    print(
        f"EXAMPLE {i + 1} | {label} | "
        f"Predicted Default Probability: "
        f"{ex['predicted_default_probability']:.1%}"
    )
    print("=" * 70)

    print("\n[CUSTOMER]")
    print(explain_with_llm(ex, audience="customer"))

    print("\n[ANALYST]")
    print(explain_with_llm(ex, audience="analyst"))