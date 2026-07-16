
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)

os.makedirs("explanations", exist_ok=True)

MODEL = "llama-3.3-70b-versatile"

FEATURE_UNITS = {
    "Age": "years",
    "Income": "annual income (currency units)",
    "LoanAmount": "loan amount requested (currency units)",
    "CreditScore": "credit score (300-850 typical range)",
    "MonthsEmployed": "months of employment history on record (not necessarily the current job)",
    "NumCreditLines": "number of open credit lines",
    "InterestRate": "interest rate (%)",
    "LoanTerm": "loan term (months)",
    "DTIRatio": "debt-to-income ratio",
    "Education": "education level",
    "EmploymentType": "employment type status",
    "MaritalStatus": "marital status",
    "HasMortgage": "has an existing mortgage",
    "HasDependents": "has dependents",
    "LoanPurpose": "stated purpose of the loan",
    "HasCoSigner": "has a co-signer",
}

SYSTEM_PROMPTS = {
    "customer": (
        "You are a helpful loan officer assistant explaining a lending decision "
        "to the loan applicant directly. Be warm, clear, and non-technical. "
        "Never mention 'SHAP', 'model', 'features', or any machine learning jargon. "
        "Speak in plain, respectful language. Keep the explanation to 3-4 sentences. "

        "CRITICAL: Only state facts using the EXACT values provided for this applicant. "
        "Never guess missing information or contradict the supplied values. "
        "For example, if HasDependents = No, say the applicant has no dependents; "
        "never imply the opposite. Mention each factor only once and do not restate "
        "the same factor using different wording. "

        "If the predicted default probability is high (roughly above 50%), explain only "
        "the top 2-3 factors increasing the applicant's risk and suggest practical ways "
        "to improve future applications. "

        "If the predicted default probability is low, confirm that the application "
        "appears relatively strong and briefly mention 1-2 positive factors (for example, "
        "stable income or good credit history) instead of discussing risk factors. "

        "Only describe this applicant's situation. "
        "Do not infer general lending rules or banking policies from the values."
    ),

    "analyst": (
        "You are an AI assistant helping a bank risk analyst quickly interpret a "
        "credit risk model's output for a specific applicant. Be precise and concise. "

        "Always cite the applicant's ACTUAL VALUE for every factor you mention, "
        "not just the feature name (for example: 'Age = 18 years' instead of "
        "'age-related risk'). "

        "Respond using exactly this structure:\n\n"
        "Verdict:\n"
        "Risk Drivers:\n"
        "Recommendation:\n\n"

        "For each risk driver include:\n"
        "- Feature name\n"
        "- Actual feature value\n"
        "- Whether it increased or decreased the predicted risk for THIS applicant\n\n"

        "Only describe this applicant. "
        "Do not infer general lending rules or invent unsupported explanations. "
        "Keep the response under 120 words."
    ),
}

def build_user_prompt(explanation: dict) -> str:
    proba = explanation["predicted_default_probability"]
    contributions = explanation["top_contributions"]

    lines = [
        
        f"Predicted default probability: {proba:.1%}",
        f"Base (average) default rate: {explanation['base_value']:.1%}",
        "",
        "IMPORTANT:",
        "- Explain only this applicant.",
        "- Do NOT assume that higher or lower values are always better or worse.",
        "- Do NOT invent lending rules.",
        "- Use the SHAP impacts only to explain this prediction.",
        "",
        "Factors for this applicant:",
    ]
    
    for c in contributions:
        unit = FEATURE_UNITS.get(c["feature"], "")
        direction = "increases risk" if c["shap_impact"] > 0 else "decreases risk"
        lines.append(
            f"- {c['feature']} = {c['value']} ({unit}): SHAP impact {c['shap_impact']:+.3f} "
            f"({direction})"
        )

    return "\n".join(lines)


def explain_with_llm(explanation: dict, audience: str = "customer") -> str:
    assert audience in SYSTEM_PROMPTS, "audience must be 'customer' or 'analyst'"
    
    try:
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[audience]},
                {"role": "user", "content": build_user_prompt(explanation)},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during Groq API call: {e}")
        return "Explanation could not be generated."

if __name__ == "__main__":
    with open("outputs/shap_examples.json") as f:
        examples = json.load(f)

    # Run both audiences on the first example (a defaulted case) as a demo
    example = examples[0]
    print("=" * 60)
    print("RAW SHAP DATA")
    print("=" * 60)
    print(json.dumps(example, indent=2))

    print("\n" + "=" * 60)
    print("CUSTOMER-FACING EXPLANATION")
    print("=" * 60)

    customer_text = explain_with_llm(
        example,
        audience="customer"
    )

    print(customer_text)

    print("\n" + "=" * 60)
    print("ANALYST-FACING EXPLANATION")
    print("=" * 60)

    analyst_text = explain_with_llm(
        example,
        audience="analyst"
    )

    print(analyst_text)

    with open("explanations/customer_explanation.txt", "w") as f:
        f.write(customer_text)

    with open("explanations/analyst_explanation.txt", "w") as f:
        f.write(analyst_text)

    print("\nSaved:")
    print("- explanations/customer_explanation.txt")
    print("- explanations/analyst_explanation.txt")